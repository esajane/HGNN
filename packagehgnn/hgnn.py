import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

from utils import local_features as lf
from utils import global_features as gf
from utils import integrate_features as inf
from utils import gated_integration as gi
from multihead import MultiHeadAttention
from global_features import GlobalInformation

class HGNN(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, num_hops, num_heads):
        super(HGNN, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.num_hops = num_hops
        self.num_heads = num_heads

        self.multi_head_attention = MultiHeadAttention(in_features, out_features, num_heads)
        self.global_feature_transform = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.global_feature_transform.weight)
        
        self.dropout = nn.Dropout(p=0.7)
        self.weight_transform = nn.Linear(2 * out_features, out_features)

    def forward(self, node_features, edge_index,):
        local_features = self.multi_head_attention(node_features, edge_index)
        global_features = gf(node_features, edge_index, self.num_nodes, self.num_hops)
        global_features = self.global_feature_transform(global_features)
        
        local_features = self.dropout(local_features)
        global_features = self.dropout(global_features)

        integrated_features = inf(local_features, global_features, self.weight_transform)
        return integrated_features


dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
model = HGNN(num_nodes=data.num_nodes, in_features=data.num_features, out_features=10, num_hops=2, num_heads=2)
optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) #important
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

for epoch in range(150):
    loss = train()
    scheduler.step() 
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

