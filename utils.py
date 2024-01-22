import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, X):
        preds = self.linear(X)
        return preds