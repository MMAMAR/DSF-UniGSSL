import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.utils import degree, to_undirected, dropout_adj, to_networkx
from torch_scatter import scatter
import networkx as nx


########### Encoder ###########
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x
    


########### Data Augmentation #############
def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        x = (1 - damp) * x + damp * agg_msg
    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    #x = nx.eigenvector_centrality_numpy(graph, max_iter=1000)
    x = nx.eigenvector_centrality(graph, max_iter=1000)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[drop_mask] = 0.

    return x


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())


def drop_edge(data, device, drop_scheme: str, drop_edge_rate: float):
    if drop_scheme == 'uniform':
        return dropout_adj(data.edge_index, p=drop_edge_rate)[0]
    
    elif drop_scheme == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
        return drop_edge_weighted(data.edge_index, drop_weights, p=drop_edge_rate, threshold=0.7)
    
    elif drop_scheme == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
        return drop_edge_weighted(data.edge_index, drop_weights, p=drop_edge_rate, threshold=0.7)
    
    elif drop_scheme == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        return drop_edge_weighted(data.edge_index, drop_weights, p=drop_edge_rate, threshold=0.7)
    
    else:
        raise Exception(f'undefined drop scheme: {drop_scheme}')


def drop_feature_global(data, device, drop_scheme: str, drop_feature_rate: float, dense: bool = False):
    if drop_scheme == 'uniform':
        return drop_feature(data.x, drop_feature_rate)
    
    elif drop_scheme == 'degree':

        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1], num_nodes=data.num_nodes)
        if dense == True:
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        return drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate)
    
    elif drop_scheme == 'pr':
        node_pr = compute_pr(data.edge_index)
        if dense == True:
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
        return drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate)
   
    elif drop_scheme == 'evc':
        node_evc = eigenvector_centrality(data)
        if dense == True:
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
        return drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate)
    
    else:
        raise Exception(f'undefined drop scheme: {drop_scheme}')


# from torch_geometric.datasets import Planetoid
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# data = dataset[0]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = data.to(device)

# # drop_scheme = ['uniform', 'degree', 'evc', 'pr']
# edge_index_1 = drop_edge(data, device, 'degree', 0.1)
# edge_index_2 = drop_edge(data, device, 'degree', 0.2)
# x_1 = drop_feature_global(data, device, 'degree', 0.1)
# x_2 = drop_feature_global(data, device, 'degree', 0.2)




def shuffle(x):
    idx = torch.randperm(x.shape[0])
    return x[idx]

# test
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# t = torch.Tensor([[1,2,3,7], [4,5,6,8], [0,9,10,11]]).to(device)
# shuffle(t)


# def shuffle_values(x, shuffle_prob=0.1, seed=129):
#     if seed is not None:
#         torch.manual_seed(seed)

#     # Get the number of rows and columns in the tensor
#     rows, cols = x.shape

#     # Iterate over each row and shuffle its values with probability shuffle_prob
#     for i in range(rows):
#         if torch.rand(1).item() < shuffle_prob:
#             x[i] = x[i][torch.randperm(cols)]

#     return x

# t = torch.Tensor([[1,2,3,7], [4,5,6,8], [0,9,10,11], [1,2,3,7]]).to(device)
# shuffle_values(t)


# def shuffle_subset(x, subset_probability=0.2):
#     torch.manual_seed(129)
    
#     # Determine the subset size based on the probability
#     subset_size = int(x.shape[0] * subset_probability)
    
#     # Check if there are enough rows for a subset
#     if subset_size > 1:
#         # Determine the subset indices
#         subset_indices = torch.randperm(x.shape[0])[:subset_size]
        
#         # Shuffle only the selected subset
#         shuffled_subset = x[subset_indices]
#         shuffled_subset = shuffled_subset[torch.randperm(subset_size)]
        
#         # Update the original tensor with the shuffled subset
#         x[subset_indices] = shuffled_subset
    
#     return x

# t = torch.Tensor([[1,2,3,7], [4,5,6,8], [0,9,10,11], [12,22,23,27]]).to(device)
# shuffle_subset(t, 0.4)
