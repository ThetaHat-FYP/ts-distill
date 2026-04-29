from ts_distill.trainer.trainer.base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Simple trainer with callback support.
    """
    
    def train_epoch(self, dataloader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            seq_len = batch.shape[1] // 2
            inputs  = batch[:, :seq_len, :]
            targets = batch[:, seq_len:, :]
            
            # Predict using only past data
            output = self.model(inputs)
            
            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def fit(self, dataloader, epochs: int, callbacks=None):
        """Main training loop with callbacks."""
        if callbacks is None:
            callbacks = []
        
        # Call on_train_begin
        for callback in callbacks:
            callback.on_train_begin(self.model)
        
        # Training loop
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            
            # Call on_epoch_end
            for callback in callbacks:
                callback.on_epoch_end(self.model, epoch, avg_loss)
        
        # Call on_train_end
        for callback in callbacks:
            callback.on_train_end(self.model)