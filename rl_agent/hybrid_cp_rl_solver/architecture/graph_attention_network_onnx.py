import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum

SQRT_TWO = float(torch.sqrt(torch.tensor(2.0)))

def scatter_softmax_onnx_safe(src, index):
    """
    ONNX-compatible scatter-based softmax.
    Computes a softmax over src values grouped by the index tensor.
    
    Args:
        src: Tensor of shape [E] or [E, D]
        index: LongTensor of shape [E], indicating group indices

    Returns:
        softmax_result: Tensor of same shape as src
    """

    if src.dim() == 1:
        src = src.unsqueeze(1)
        squeeze_output = True
    else:
        squeeze_output = False

    expanded_index = index.unsqueeze(-1).expand_as(src)

    # Proper initialization for max
    max_per_index = torch.full_like(src, float('-inf')).scatter_reduce(
                        0, expanded_index, src, reduce="amax"
                    )
    gathered_max = max_per_index[index]

    exp = torch.exp(src - gathered_max)

    sum_per_index = torch.zeros_like(src).scatter_reduce(
                        0, expanded_index, exp, reduce="sum"
                    )
    gathered_sum = sum_per_index[index]

    softmax = exp / (gathered_sum + 1e-16)

    if squeeze_output:
        softmax = softmax.squeeze(1)

    return softmax


class EdgeFtLayerONNX(nn.Module):
    def __init__(self, v_in_dim, v_out_dim, e_in_dim, e_out_dim):
        super().__init__()
        self.W_a = nn.Parameter(torch.empty((2 * v_in_dim + e_in_dim, v_out_dim)))
        self.W_T = nn.Parameter(torch.empty((2 * v_in_dim + e_in_dim, v_out_dim)))
        self.b_T = nn.Parameter(torch.zeros(v_out_dim))

        self.W_e = nn.Parameter(torch.empty((v_in_dim, e_out_dim)))
        self.W_ee = nn.Parameter(torch.empty((e_in_dim, e_out_dim)))

        self.prelu = nn.PReLU()

        nn.init.xavier_normal_(self.W_a, gain=SQRT_TWO)
        nn.init.xavier_normal_(self.W_T, gain=SQRT_TWO)
        nn.init.xavier_normal_(self.W_e, gain=SQRT_TWO)
        nn.init.xavier_normal_(self.W_ee, gain=SQRT_TWO)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the layer.

        x (num_nodes, v_in_dim)        : Node features 
        edge_index (2, num_edges)      : Edge indices. [[src1, src2, ...], [dst1, dst2, ...]]
        edge_attr (num_edges, e_in_dim): Edge features

        Returns:
            new_x: Updated node features (num_nodes, v_out_dim)
            new_e_feat: Updated edge features (num_edges, e_out_dim)
        """

        src, dst = edge_index[0], edge_index[1]
        N1 = x[src]  # (num_edges, v_in_dim) Node features of all the source nodes
        N2 = x[dst]  # (num_edges, v_in_dim) Node features of all the destination nodes
        e = edge_attr  # (num_edges, e_in_dim)

        # Update edge features
        new_e_feat = N1 @ self.W_e + N2 @ self.W_e + e @ self.W_ee

        # Create messages and compute attention logits
        cat = torch.cat([N2, e, N1], dim=1)  # (num_edges, 2*v_in_dim + e_in_dim)
        attention_logits = self.prelu(cat @ self.W_a)  # (num_edges, v_out_dim)
        messages = cat @ self.W_T  # (num_edges, v_out_dim)

        attention = scatter_softmax_onnx_safe(attention_logits, dst)  # (num_edges, v_out_dim)

        # Prepare for per-dst-node softmax
        num_nodes = x.size(0)
        v_out_dim = messages.size(1)

        agg_msg = attention * messages # (num_edges, v_out_dim)

        # Aggregate messages per destination node
        index = dst.unsqueeze(1).expand(-1, v_out_dim)  # (num_edges, v_out_dim)
        new_x = torch.zeros((num_nodes, v_out_dim), device=x.device) # (num_nodes, v_out_dim)
        new_x = new_x.scatter_add(0, index, agg_msg) # (num_nodes, v_out_dim)

        new_x += self.b_T
        return new_x, new_e_feat



class GATNetworkONNX(nn.Module):
    def __init__(self, layer_features, n_hidden_layer, latent_dim, output_dim, graph_pooling=False):
        super().__init__()
        self.graph_pooling = graph_pooling

        self.layers = nn.ModuleList([
            EdgeFtLayerONNX(v_in, v_out, e_in, e_out)
            for (v_in, e_in), (v_out, e_out) in zip(layer_features[:-1], layer_features[1:])
        ])

        last_dim = layer_features[-1][0]
        fcs = [nn.Linear(last_dim, latent_dim)]
        for _ in range(n_hidden_layer):
            fcs.append(nn.Linear(latent_dim, latent_dim))
        self.fc_layer = nn.Sequential(*fcs)
        self.fc_out = nn.Linear(latent_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            edge_attr = F.relu(edge_attr)

        if self.graph_pooling:
            out = torch.max(x, dim=0)[0]
            out = self.fc_layer(out)
            out = self.fc_out(out)
            return out
        else:
            x = self.fc_layer(x)
            x = self.fc_out(x)
            return x
