import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.per_head_features = out_features // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(in_features, self.per_head_features, bias=False) for _ in range(num_heads)])
        self.merge_layer = nn.Linear(out_features, out_features)

    def forward(self, node_features, edge_index):
        head_outputs = []
        for i in range(self.num_heads):
            transformed = self.linears[i](node_features)
            source = transformed[edge_index[0]]
            target = transformed[edge_index[1]]
            attention_scores = (source * target).sum(-1)
            attention_scores = F.softmax(attention_scores, dim=0)
            output = torch.zeros_like(transformed)
            output.index_add_(0, edge_index[0], attention_scores.unsqueeze(-1) * target)
            head_outputs.append(output)
        
        concatenated = torch.cat(head_outputs, dim=1)
        return self.merge_layer(concatenated)
