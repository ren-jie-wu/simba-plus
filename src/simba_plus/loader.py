import torch
from torch.utils.data import Dataset
from torch_geometric.typing import EdgeType


class CustomIndexDataset(Dataset):
    def __init__(
        self,
        edge_attr_type: EdgeType,
        index: torch.Tensor,
    ):
        self.edge_attr_type = edge_attr_type
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.edge_attr_type, self.index[idx]
