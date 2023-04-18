import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, num_features, embedding_dim = 8, hidden_dim=64):
        super(DeepFM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # linear
        self.linear = nn.Linear(num_features, 1)
        # FM
        self.fm_embedding = nn.Embedding(num_features, self.embedding_dim)
        # Deep
        self.deep_embedding = nn.Embedding(num_features, self.hidden_dim)
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
        # linear
        linear_output = self.linear(x)
        # FM
        # 公式中的<w, x>
        fm_embedding = self.fm_embedding(x)
        # 公式中的第二部分用数学技巧展开后可以由下面两部分的和构成
        fm_embedding_sum = torch.sum(fm_embedding, dim=1)
        fm_embedding_square = torch.sum(fm_embedding * fm_embedding, dim=1)
        fm_output = fm_embedding + fm_embedding_sum + fm_embedding_square
        # Deep
        deep_embedding = self.deep_embedding(x)
        deep_output = self.deep_layers(torch.sum(deep_embedding, dim=1))
        # Output
        output = linear_output + fm_output + deep_output
        output = torch.sigmoid(output)
        return output
