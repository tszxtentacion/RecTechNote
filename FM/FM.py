import torch.nn as nn
import torch


class FM(nn.Module):
    def __init__(self, n, k):
        super(FM, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(n, 1)
        self.v = nn.Parameter(torch.randn(n, k))

    def forward(self, x):
        linear_part = self.linear(x).squeeze()
        interaction_part1 = torch.matmul(x, self.v)
        interaction_part2 = torch.matmul(x ** 2, self.v ** 2)
        interaction_part = (interaction_part1 ** 2 - interaction_part2).sum(1)
        y_hat = linear_part + 0.5 * interaction_part
        return y_hat
