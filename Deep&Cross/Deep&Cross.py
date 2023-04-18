import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepAndCross(nn.Module):
    def __init__(self, field_dims, embed_dim, num_layers, dropout):
        """
        field_dims: 每个字段的取值个数，用于 Embedding 层的输入维度
        embed_dim: Embedding 层的输出维度
        num_layers: Deep 层的层数
        dropout: Dropout 的概率
        """
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Embedding 层
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)

        # Deep 层
        self.deep_layers = nn.ModuleList([nn.Linear(sum(field_dims), sum(field_dims)) for _ in range(num_layers)])
        self.deep_dropout = nn.Dropout(dropout)

        # Cross 层
        # 每一层交叉层的权重：num_fields * embedding_dim
        self.cross_weights = nn.ParameterList([nn.Parameter(torch.randn(sum(field_dims))) for _ in range(num_layers)])

        # 输出层
        self.fc = nn.Linear(embed_dim + sum(field_dims), 1)

    def forward(self, x):
        """
        x: 输入特征张量，维度为(batch_size, num_fields)
        """
        # Embedding 层
        emb = self.embedding(x)  # (batch_size, num_fields, embed_dim)
        emb = emb.flatten(start_dim=1)  # (batch_size, num_fields * embed_dim)

        # Deep 层
        deep_input = emb
        for i in range(self.num_layers):
            deep_input = self.deep_layers[i](deep_input)
            deep_input = F.relu(deep_input)
            deep_input = self.deep_dropout(deep_input)

        # Cross 层
        cross_input = emb
        for i in range(self.num_layers):
            cross_weights = self.cross_weights[i]
            cross_input = torch.einsum('bnm,bm->bn', cross_input, cross_weights)[:, :, None] * cross_input
            cross_input = cross_input + emb

        # 输出层
        output = torch.cat([emb, deep_input, cross_input], dim=1)
        output = self.fc(output).squeeze()
        output = torch.sigmoid(output)

        return output
