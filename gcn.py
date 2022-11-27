import torch
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 15)
        self.conv2 = GCNConv(15, 20)
        self.conv3 = GCNConv(20, 27)
        self.conv4 = GCNConv(27, 36)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).selu()
        x = self.conv2(x, edge_index).selu()
        x = self.conv3(x, edge_index).selu()
        x = self.conv4(x, edge_index).selu()
        return x
