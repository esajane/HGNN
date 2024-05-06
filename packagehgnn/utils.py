import torch
import torch.nn.functional as F

def local_features(node_features, edge_index, linear_weights, attention_weights):
    """
    Args:
    node_features (torch.Tensor): Node features of shape (num_nodes, in_features).
    edge_index (torch.LongTensor): Edge indices of shape (2, num_edges).
    linear_weights (torch.Tensor): Weights for the linear transformation (in_features, out_features).
    attention_weights (torch.Tensor): Weights for the attention mechanism (2 * out_features).

    Returns:
    torch.Tensor: Aggregated node features of shape (num_nodes, out_features).
    """
    transformed_features = F.linear(node_features, linear_weights)
    source_features = transformed_features[edge_index[0]]
    target_features = transformed_features[edge_index[1]]
    edge_features = torch.cat([source_features, target_features], dim=1)
    attention_scores = F.leaky_relu(torch.matmul(edge_features, attention_weights))
    attention_scores = F.softmax(attention_scores.squeeze(), dim=0)
    node_out_features = torch.zeros_like(transformed_features)
    for i in range(edge_index.size(1)):
        node_out_features[edge_index[0][i]] += attention_scores[i] * target_features[i]
    return node_out_features


def expand_neighborhood(edge_index, num_nodes, num_hops):
    indices = edge_index
    values = torch.ones(indices.shape[1], dtype=torch.float32)
    adjacency_matrix = torch.sparse_coo_tensor(indices, values, size=[num_nodes, num_nodes])
    hop_matrix = torch.eye(num_nodes)
    for _ in range(num_hops):
        hop_matrix = torch.sparse.mm(adjacency_matrix, hop_matrix)
    
    return hop_matrix

def uniform_aggregation(node_features, hop_matrix):
    gathered_features = torch.sparse.mm(hop_matrix, node_features)
    degree_matrix = hop_matrix.sum(dim=1).unsqueeze(1)  
    averaged_features = gathered_features / degree_matrix
    
    return averaged_features

def global_features(node_features, edge_index, num_nodes, num_hops):
    """
    Args: 
    node_features (torch.Tensor): Node features of shape (num_nodes, in_features).
    edge_index (torch.LongTensor): Edge indices of shape (2, num_edges).
    num_nodes (int): Number of nodes in the graph.
    num_hops (int): Number of hops to expand the neighborhood.

    Returns:
    torch.Tensor: Global features of shape (num_nodes, in_features).
    """
    hop_matrix = expand_neighborhood(edge_index, num_nodes, num_hops)
    global_features = uniform_aggregation(node_features, hop_matrix)
    
    return global_features

def integrate_features(local_features, global_features, weight_transform):
    """
    Args:
    local_features (torch.Tensor): Local features of shape (num_nodes, out_features).
    global_features (torch.Tensor): Global features of shape (num_nodes, out_features).
    weight_transform (torch.nn.Linear): Linear layer to transform concatenated features.

    Returns:
    torch.Tensor: Integrated features of shape (num_nodes, out_features).
    """
    concatenated_features = torch.cat([local_features, global_features], dim=1)
    logits = weight_transform(concatenated_features)  # Outputs logits for each class
    return logits


def gated_integration(local_features, global_features, gate_weights, transform_weights):
    """
    Args:
    local_features (torch.Tensor): Local features of shape (num_nodes, out_features).
    global_features (torch.Tensor): Global features of shape (num_nodes, out_features).
    gate_weights (torch.Tensor): Weights for the gating mechanism (out_features, 2 * out_features).
    transform_weights (torch.Tensor): Weights for the linear transformation (out_features, 2 * out_features).

    Returns:
    torch.Tensor: Integrated features of shape (num_nodes, out_features).
    """
    concatenated_features = torch.cat([local_features, global_features], dim=1)
    gating_weights = torch.sigmoid(F.linear(concatenated_features, gate_weights))
    transformed_features = F.linear(concatenated_features, transform_weights)
    
    integrated_features = gating_weights * transformed_features
    return integrated_features
