import torch
import torch.nn as nn
import torch.optim as optim

from ts_distill.models import MLPModel
from ts_distill.runner import Trainer
# 10 samples, 6 features each
x = torch.randn(10, 6)
y = torch.randn(10, 1)
model = MLPModel(
    input_dim=6,
    hidden_dim=8,
    output_dim=1
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn
)
for epoch in range(5):
    loss = trainer.train_step(x, y)
    print(f"Epoch {epoch}, Loss: {loss}")
