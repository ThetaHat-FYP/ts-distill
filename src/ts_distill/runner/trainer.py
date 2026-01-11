import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_step(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        predictions = self.model(x)
        loss = self.loss_fn(predictions, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()
