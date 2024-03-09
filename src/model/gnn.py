import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from src.model.gnn_layer.gat_layer import GATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4, link_prediction=False):
        super(GAT, self).__init__()
        self.link_prediction = link_prediction
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x1, edge_index1, x2, edge_index2, edge_attr=None):            #  edge_index1: torch.Size([2, 9196])
        # print(f"Max index in edge_index1: {edge_index1.max()}")                     # 253
        # print(f"Min index in edge_index1: {edge_index1.min()}")
        # print("X1: ", x1)
        # print(f"Number of nodes (x1 size 0): {x1.size(0)}")                         # 4
        # if edge_index1.max() >= x1.size(0):
        #     raise RuntimeError("Invalid edge_index1 detected")

        for i, conv in enumerate(self.convs[:-1]):
            x1, edge_attr = conv(x1, edge_index=edge_index1, edge_attr=edge_attr, )
            x1 = self.bns[i](x1)
            x1 = F.relu(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1, edge_attr = self.convs[-1](x1, edge_index=edge_index1, edge_attr=edge_attr)

        for i, conv in enumerate(self.convs[:-1]):
            x2, edge_attr = conv(x2, edge_index=edge_index2, edge_attr=edge_attr, )
            x2 = self.bns[i](x2)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2, edge_attr = self.convs[-1](x2, edge_index=edge_index2, edge_attr=edge_attr)

        return x1, x2, edge_attr




load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
}
