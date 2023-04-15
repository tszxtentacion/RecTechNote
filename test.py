import torch

batch_size = 2
num_fields = 3
embed_dim = 2
num_layers = 1

cross_input = torch.tensor([
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
])

cross_weights = torch.tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
