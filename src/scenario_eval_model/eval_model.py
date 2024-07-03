import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.nn as pyg_nn

class GNNRegressionModel(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, weather_input_dim, hidden_dim, output_dim):
        super(GNNRegressionModel, self).__init__()
        
        self.node_gcn1 = GCNConv(node_input_dim, hidden_dim)
        self.node_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.edge_gcn1 = GCNConv(edge_input_dim, hidden_dim)
        self.edge_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.weather_mlp1 = nn.Linear(weather_input_dim, hidden_dim)
        self.weather_mlp2 = nn.Linear(hidden_dim, hidden_dim)

        self.readout_mlp1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.readout_mlp2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features, edge_attr, weather_attr, edge_index, node_batch, edge_batch):

        node_h = F.relu(self.node_gcn1(node_features, edge_index))
        node_h = F.dropout(node_h, p=0.5, training=self.training)
        node_h = self.node_gcn2(node_h, edge_index)
        
        edge_h = F.relu(self.edge_gcn1(edge_attr, edge_index))
        edge_h = F.dropout(edge_h, p=0.5, training=self.training)
        edge_h = self.edge_gcn2(edge_h, edge_index)
        
        weather_h = F.relu(self.weather_mlp1(weather_attr,))
        weather_h = self.weather_mlp2(weather_h)

        # Use global_mean_pool to aggregate node and edge features for each graph in the batch
        pooled_node_h = global_mean_pool(node_h,node_batch)
        pooled_edge_h = global_mean_pool(edge_h, edge_batch)
        combined_h = torch.cat([pooled_node_h, pooled_edge_h, weather_h], dim=-1)
        out_h = F.relu(self.readout_mlp1(combined_h))
        out = self.readout_mlp2(out_h)

        return out

class GNNBinaryClassificationModel(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, weather_input_dim, hidden_dim):
        super(GNNBinaryClassificationModel, self).__init__()
        
        self.node_gcn1 = GCNConv(node_input_dim, hidden_dim)
        self.node_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.edge_gcn1 = GCNConv(edge_input_dim, hidden_dim)
        self.edge_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        self.weather_mlp1 = nn.Linear(weather_input_dim, hidden_dim)
        self.weather_mlp2 = nn.Linear(hidden_dim, hidden_dim)

        self.readout_mlp1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.readout_mlp2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, edge_attr, weather_attr, edge_index, node_batch, edge_batch):

        node_h = F.relu(self.node_gcn1(node_features, edge_index))
        node_h = F.dropout(node_h, p=0.5, training=self.training)
        node_h = self.node_gcn2(node_h, edge_index)
        
        edge_h = F.relu(self.edge_gcn1(edge_attr, edge_index))
        edge_h = F.dropout(edge_h, p=0.5, training=self.training)
        edge_h = self.edge_gcn2(edge_h, edge_index)
        
        weather_h = F.relu(self.weather_mlp1(weather_attr,))
        weather_h = self.weather_mlp2(weather_h)

        # Use global_mean_pool to aggregate node and edge features for each graph in the batch
        pooled_node_h = global_mean_pool(node_h,node_batch)
        pooled_edge_h = global_mean_pool(edge_h, edge_batch)
        combined_h = torch.cat([pooled_node_h, pooled_edge_h, weather_h], dim=-1)
        out_h = F.relu(self.readout_mlp1(combined_h))
        out = self.readout_mlp2(out_h)
        out = self.sigmoid(out)

        return out

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean',device='cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1-alpha, alpha]).to(device)
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha).to(device)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        targets_index =targets.long()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1-pt)**self.gamma * BCE_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets_index] * focal_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        else:
            return torch.sum(focal_loss)



class ImprovedGNNBinaryClassificationModel(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, weather_input_dim, hidden_dim, num_heads=1):
        super(ImprovedGNNBinaryClassificationModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # ...

        self.node_gat1 = pyg_nn.GATConv(node_input_dim, self.hidden_dim, heads=self.num_heads)
        self.node_gat2 = pyg_nn.GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=self.num_heads)
        
        self.edge_gat1 = pyg_nn.GATConv(edge_input_dim, self.hidden_dim, heads=self.num_heads)
        self.edge_gat2 = pyg_nn.GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=self.num_heads)
        
        self.weather_mlp1 = nn.Linear(weather_input_dim, self.hidden_dim)
        self.weather_mlp2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.node_bn1 = nn.BatchNorm1d(self.hidden_dim * self.num_heads)
        self.node_bn2 = nn.BatchNorm1d(self.hidden_dim * self.num_heads)
        self.edge_bn1 = nn.BatchNorm1d(self.hidden_dim * self.num_heads)
        self.edge_bn2 = nn.BatchNorm1d(self.hidden_dim * self.num_heads)
        self.weather_bn1 = nn.BatchNorm1d(self.hidden_dim)
        
        self.readout_mlp1 = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.readout_mlp2 = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, edge_attr, weather_attr, edge_index, node_batch, edge_batch):

        node_h = F.relu(self.node_gat1(node_features, edge_index))
        node_h = self.node_bn1(node_h)
        node_h = F.dropout(node_h, p=0.5, training=self.training)
        node_h = self.node_gat2(node_h, edge_index)
        node_h = self.node_bn2(node_h)
        node_h = node_h.view(-1, self.hidden_dim, self.num_heads).mean(dim=-1)

        edge_h = F.relu(self.edge_gat1(edge_attr, edge_index))
        edge_h = self.edge_bn1(edge_h)
        edge_h = F.dropout(edge_h, p=0.5, training=self.training)
        edge_h = self.edge_gat2(edge_h, edge_index)
        edge_h = self.edge_bn2(edge_h)
        edge_h = edge_h.view(-1, self.hidden_dim, self.num_heads).mean(dim=-1)

        weather_h = F.relu(self.weather_mlp1(weather_attr))
        weather_h = self.weather_bn1(weather_h)
        weather_h = self.weather_mlp2(weather_h)

        # Use global_mean_pool to aggregate node and edge features for each graph in the batch
        pooled_node_h = global_mean_pool(node_h, node_batch)
        pooled_edge_h = global_mean_pool(edge_h, edge_batch)
        combined_h = torch.cat([pooled_node_h, pooled_edge_h, weather_h], dim=-1)
        out_h = F.relu(self.readout_mlp1(combined_h))
        out = self.readout_mlp2(out_h)
        out = self.sigmoid(out)

        return out


