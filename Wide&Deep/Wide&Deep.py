import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    def __init__(self, num_features, embedding_dim=8, hidden_dim=64):
        super(WideAndDeep, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Wide
        self.wide_linear = nn.Linear(num_features, 1)

        # Deep
        self.deep_embedding = nn.Embedding(num_features, embedding_dim)
        self.deep_layers = nn.Sequential(
            nn.Linear(num_features * embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Wide
        wide_output = self.wide_linear(x)

        # Deep
        deep_embedding = self.deep_embedding(x)
        deep_embedding = deep_embedding.view(-1, x.shape[1] * self.embedding_dim)
        deep_output = self.deep_layers(deep_embedding)

        # Output
        output = wide_output + deep_output
        output = torch.sigmoid(output)
        return output
