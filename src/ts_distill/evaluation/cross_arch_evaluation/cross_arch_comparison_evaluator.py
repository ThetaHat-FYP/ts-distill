"""
Cross-Architecture Comparison Evaluator
-----------------------------------------
Given an already-trained expert trajectory, distils synthetic data with BOTH
standard MTT (param_mtt) and phase-aware MTT (phase_aware_mtt) from a shared
RNG state — so both methods start from an identical synthetic_init and the
matching loss is the only variable — then evaluates each resulting synthetic
set on every architecture in `student_archs` as the student
(cross-architecture matrix).

This holds the algorithmic core moved out of
example/experiments/experiment_matrix.py::run_cross_arch_comparison(): dual
distillation + cross-arch evaluation. Expert training and phase-boundary
detection stay in the caller (they are generic, shared with the same-arch
`run_single_experiment` pipeline and with
ts_distill.trajectory.phase_detector) — this class only owns the
"synthetic data made by one architecture, judged on every architecture"
concern, matching the split already used by
ts_distill.evaluation.hybrid_evaluation.hybrid.BaseHybridEvaluator (mixing
algorithm here, CSV I/O stays with the caller).
"""

from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from ts_distill.data_pipeline.splitter import make_windows
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.distillation_algorithm.phase_aware_mtt import (
    PhaseAwareMTTDistiller,
)
from ts_distill.distillation_core.initializer.random_sample_initializer import (
    RandomSampleInitializer,
)
from ts_distill.evaluation.base import BaseEvaluator
from ts_distill.evaluation.evaluation import Evaluator
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher

from ts_distill._logging import get_logger

logger = get_logger(__name__)


class CrossArchComparisonEvaluator(BaseEvaluator):
    """
    Args:
        model_factory: Callable(arch_name: str) -> fresh nn.Module for that
            architecture (e.g. `lambda name: create_model(model_type=name, ...)`).
        student_archs: Architectures to evaluate the synthetic data on, e.g.
            ['DLinear', 'LSTM', 'MLP', 'CNN'].
        seq_len / pred_len: Input / forecast horizon lengths.
        device: Device for expert/student Trainer objects. Distillers are
            always constructed without a device kwarg (CPU), matching the
            rest of the MTT pipeline — this keeps CPU/CUDA RNG streams from
            diverging, which would otherwise break the shared synthetic_init
            between param_mtt and phase_aware_mtt.

    A single instance is meant to be reused across all experts (and seeds)
    for one dataset — `real_mse` is cached internally per (student_arch, seed)
    since it does not depend on which expert produced the synthetic data.
    """

    def __init__(
        self,
        model_factory: Callable[[str], "torch.nn.Module"],
        student_archs: List[str],
        seq_len: int,
        pred_len: int,
        device: str = "cpu",
    ) -> None:
        self.model_factory = model_factory
        self.student_archs = student_archs
        self.seq_len        = seq_len
        self.pred_len       = pred_len
        self.window_size    = seq_len + pred_len
        self.device          = device
        self._real_mse_cache: Dict[tuple, float] = {}

    # =========================================================================
    # BaseEvaluator contract
    # =========================================================================

    def test_on_real(self, model, real_test_data, eval_batch_size: int = 32):
        evaluator = Evaluator(seq_len=self.seq_len, batch_size=eval_batch_size)
        return evaluator.test_on_real(model, real_test_data)

    # =========================================================================
    # PUBLIC — Dual-method distillation
    # =========================================================================

    def distill_both_methods(
        self,
        expert_name:    str,
        recorder,
        raw_train_data: torch.Tensor,
        val_data:       torch.Tensor,
        phase_boundary: Optional[int],
        cfg:            dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Distil param_mtt and phase_aware_mtt from the SAME expert trajectory
        and an identical synthetic_init (RNG state is reset between the two
        calls). A method that raises is skipped rather than aborting the other.

        Returns:
            {'param_mtt': Tensor, 'phase_aware_mtt': Tensor} — missing a key
            if that method's distillation failed.
        """
        post_expert_rng_state = torch.get_rng_state()
        initializer = RandomSampleInitializer()

        common_kwargs = dict(
            matcher                = MSEMatcher(),
            model_factory          = lambda: self.model_factory(expert_name),
            expert_recorder        = recorder,
            expert_epochs          = cfg["trajectory_gap"],
            syn_batch_size         = cfg["batch_size"],
            synthetic_lr           = cfg["synthetic_lr"],
            student_lr             = cfg["student_lr"],
            student_steps          = cfg["student_steps"],
            snapshot_student_steps = cfg["snapshot_student_steps"],
            seq_len                = self.seq_len,
            pred_len                = self.pred_len,
        )

        synthetic: Dict[str, torch.Tensor] = {}

        torch.set_rng_state(post_expert_rng_state)
        try:
            synthetic_init = initializer.initialize_sequence(raw_train_data, cfg["n_synthetic"])
            distiller = MTTDistiller(initializer=initializer, **common_kwargs)
            synthetic["param_mtt"] = distiller.distill(
                synthetic_init=synthetic_init, n_steps=cfg["n_distill_steps"], val_data=val_data,
            )
        except Exception as exc:
            logger.info(f"  [FAIL] param_mtt distillation: {exc}")

        torch.set_rng_state(post_expert_rng_state)
        try:
            synthetic_init = initializer.initialize_sequence(raw_train_data, cfg["n_synthetic"])
            distiller = PhaseAwareMTTDistiller(
                initializer=initializer, phase_boundary=phase_boundary, **common_kwargs,
            )
            synthetic["phase_aware_mtt"] = distiller.distill(
                synthetic_init=synthetic_init, n_steps=cfg["n_distill_steps"], val_data=val_data,
            )
        except Exception as exc:
            logger.info(f"  [FAIL] phase_aware_mtt distillation: {exc}")

        return synthetic

    # =========================================================================
    # PUBLIC — Cross-architecture evaluation matrix
    # =========================================================================

    def evaluate_cross_arch(
        self,
        expert_name:      str,
        seed:             int,
        synthetic_by_method: Dict[str, torch.Tensor],
        train_data:       torch.Tensor,
        val_data:         torch.Tensor,
        test_data:        torch.Tensor,
        phase_boundary:   Optional[int],
        n_pairs_available: int,
        cfg:              dict,
        row_done_fn:      Optional[Callable[[str, str], bool]] = None,
        on_row:           Optional[Callable[[dict], None]] = None,
    ) -> List[dict]:
        """
        Trains + evaluates every architecture in `student_archs` on real data
        (cached per (student, seed)) and on each available synthetic set.

        Args:
            row_done_fn: Optional callable(student_name, method) -> bool used
                to skip already-computed cells (resume support). When None,
                every cell is (re)computed.
            on_row: Optional callable(row_dict) invoked immediately after each
                row is computed — lets the caller append it to a CSV as it
                goes (crash-safe) instead of waiting for the full return list.

        Returns:
            List of row dicts (also the full set passed to `on_row`, in the
            same order): expert_model, student_model, seed, method,
            phase_boundary, student_lr, synthetic_lr, n_distill_steps,
            real_mse, transfer_mse, mse_ratio, n_pairs_available, notes.
            (Caller adds 'dataset' — this evaluator is dataset-agnostic.)
        """
        eval_val_loader = TorchDataLoader(
            val_data, batch_size=cfg["eval_batch_size"], shuffle=False
        )
        rows: List[dict] = []

        for student_name in self.student_archs:

            real_mse = self._get_real_mse(
                student_name, seed, train_data, eval_val_loader, test_data, cfg,
            )

            for method, synthetic_seq in synthetic_by_method.items():

                if row_done_fn is not None and row_done_fn(student_name, method):
                    continue

                row_base = {
                    "expert_model":       expert_name,
                    "student_model":      student_name,
                    "seed":               seed,
                    "method":             method,
                    "phase_boundary":     phase_boundary if method == "phase_aware_mtt" else "",
                    "student_lr":         cfg["student_lr"],
                    "synthetic_lr":       cfg["synthetic_lr"],
                    "n_distill_steps":    cfg["n_distill_steps"],
                    "n_pairs_available":  n_pairs_available,
                }

                try:
                    transfer_mse = self._eval_synthetic(
                        student_name, synthetic_seq, eval_val_loader, test_data, cfg,
                    )
                except Exception as exc:
                    logger.info(f"    [FAIL] {method} synthetic eval ({student_name}): {exc}")
                    row = {**row_base, "real_mse": real_mse,
                           "notes": f"synthetic_eval_failed: {str(exc)[:120]}"}
                    rows.append(row)
                    if on_row is not None:
                        on_row(row)
                    continue

                mse_ratio = (
                    transfer_mse / real_mse
                    if (real_mse and real_mse == real_mse and real_mse > 0)  # real_mse == real_mse rules out NaN
                    else float("nan")
                )

                row = {
                    **row_base,
                    "real_mse":     real_mse,
                    "transfer_mse": transfer_mse,
                    "mse_ratio":    mse_ratio,
                    "notes":        "",
                }
                rows.append(row)
                if on_row is not None:
                    on_row(row)

        return rows

    # =========================================================================
    # PRIVATE — helpers
    # =========================================================================

    def _get_real_mse(
        self, student_name, seed, train_data, eval_val_loader, test_data, cfg,
    ) -> float:
        key = (student_name, seed)
        if key in self._real_mse_cache:
            return self._real_mse_cache[key]

        logger.info(f"    Computing real baseline for {student_name}...")
        torch.manual_seed(0)
        model   = self.model_factory(student_name).to(self.device)
        trainer = Trainer(
            model=model, optimizer=torch.optim.Adam(model.parameters(), lr=cfg["eval_lr"]),
            criterion=torch.nn.MSELoss(), device=self.device, seq_len=self.seq_len,
        )
        try:
            trainer.fit(
                dataloader=TorchDataLoader(train_data, batch_size=cfg["eval_batch_size"], shuffle=True),
                epochs=cfg["eval_max_epochs"], val_loader=eval_val_loader,
                patience=cfg["early_stop_patience"],
            )
            mse = self.test_on_real(
                model, test_data.to(self.device), eval_batch_size=cfg["eval_batch_size"],
            )["MSE"]
        except Exception as exc:
            logger.info(f"    [FAIL] real eval ({student_name}): {exc}")
            mse = float("nan")

        self._real_mse_cache[key] = mse
        return mse

    def _eval_synthetic(
        self, student_name, synthetic_seq, eval_val_loader, test_data, cfg,
    ) -> float:
        syn_windows = torch.tensor(
            make_windows(synthetic_seq.cpu().numpy(), self.window_size), dtype=torch.float32,
        )
        torch.manual_seed(1)
        model   = self.model_factory(student_name).to(self.device)
        trainer = Trainer(
            model=model, optimizer=torch.optim.Adam(model.parameters(), lr=cfg["eval_lr"]),
            criterion=torch.nn.MSELoss(), device=self.device, seq_len=self.seq_len,
        )
        trainer.fit(
            dataloader=TorchDataLoader(syn_windows, batch_size=cfg["eval_batch_size"], shuffle=True),
            epochs=cfg["eval_max_epochs"], val_loader=eval_val_loader,
            patience=cfg["early_stop_patience"],
        )
        return self.test_on_real(
            model, test_data.to(self.device), eval_batch_size=cfg["eval_batch_size"],
        )["MSE"]
