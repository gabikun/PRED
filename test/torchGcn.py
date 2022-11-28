import torch
import torch.nn.functional as F
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
        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = self.conv2(x, edge_index)
        x = F.selu(x)
        x = self.conv3(x, edge_index)
        x = F.selu(x)
        x = self.conv4(x, edge_index)
        x = F.selu(x)
        return x

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        # On ne peut pas faire comme ça : data = dataset[0].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            # On ne peut pas faire comme ça : out = model(data)
            # Idem loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            # Idem loss.backward()
            optimizer.step()