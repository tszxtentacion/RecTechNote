import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from preprocess_data import df_train
from CTRModel import model
import numpy as np


class CTRDataset(Dataset):
    def __init__(self, df_train):
        self.y = df_train["clicked"].values
        self.X = df_train.drop("clicked", axis=1).values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CTRDataset(df_train)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_idx, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred.view(-1), y.float())
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
