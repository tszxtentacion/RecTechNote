import torch
import torch.nn as nn
from preprocess_data import df_train, categorical_cols, continuous_cols


class CTRModel(nn.Module):
    def __init__(self, categorical_cols_, continuous_cols_, embedding_dim=10):
        super().__init__()
        # 每种类别型特征都被embed成一个维度为embedding_dim的向量, 每个类别型特征组合起来，为这样一个list
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings=len(df_train[col].unique()),
                                                      embedding_dim=embedding_dim) for col in categorical_cols_])
        # 下方的输入维度数：连续性+类别型*embed后的维度 （一个类型的特征被embed后从one-hot向量变成了embedding_dim维度的向量）
        self.linear = nn.Linear(in_features=len(continuous_cols_) + len(categorical_cols_) * embedding_dim, out_features=1)

    def forward(self, x):
        categorical_data = [self.embeddings[i](int(x[:, i])) for i in range(len(self.embeddings))]
        # 从ModuleList中取出之后，进行拼接，转换为一个长向量
        categorical_data = torch.cat(categorical_data, dim=1)
        continuous_data = x[:, len(self.embeddings):]
        # 将类别型特征、连续型特征进行拼接，拼成一个长向量
        x = torch.cat([categorical_data, continuous_data], dim=1)
        # 下方可以根据实际模型添加不同的网络构件
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


model = CTRModel(categorical_cols_=categorical_cols, continuous_cols_=continuous_cols)