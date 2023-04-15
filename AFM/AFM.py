import torch
import torch.nn as nn
import torch.nn.functional as F


class AFM(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields, attention_factor=4, dropout_rate=0.2):
        super(AFM, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields    # 特征分组的数量，通常等于特殊类型的数量（如用户的年龄、性别、地域信息为一个组）
        self.attention_factor = attention_factor

        # 定义一阶特征的embedding层
        self.first_order_embeddings = nn.ModuleList([
            nn.Embedding(num_features[i], 1) for i in range(num_fields)
        ])
        # 定义二阶embedding层
        self.second_order_embeddings = nn.ModuleList([
            nn.Embedding(num_features[i], embedding_dim) for i in range(num_fields)
        ])

        # attention层
        self.attention_W = nn.Parameter(torch.randn(embedding_dim, attention_factor))

        # 输出层（一阶特征 + 二阶特征拼接）
        self.output_layer = nn.Linear(embedding_dim + 1, 1)

        # dropout层
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch_size = x.shape[0]

        # 一阶特征
        first_order_output = torch.zeros(batch_size, 1)
        for i in range(self.num_fields):
            first_order_output += self.first_order_embeddings[i](x[:, i].unsqueeze(1))

        # 二阶特征
        second_order_output = torch.zeros(batch_size, self.embedding_dim)
        for i in range(self.num_fields):
            for j in range(self.num_fields):
                second_order_output += self.second_order_embeddings[i](x[:, i].unsqueeze(1)) \
                                * self.second_order_embeddings[i](x[:, j].unsqueeze(1))
        # attention层处理后的二阶特征
        attention_scores = F.softmax(torch.matmul(second_order_output, self.attention_W), dim=1)
        second_order_output = torch.sum(attention_scores * second_order_output, dim=1)

        # 拼接特征（一阶特征 + attention加权后的二阶特征）
        output = torch.cat([first_order_output, second_order_output], dim=1)

        # dropout
        output = self.dropout_layer(output)

        # 输出层
        output = self.output_layer(output)

        return output

