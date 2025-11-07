import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_cluster import random_walk

from torch_geometric.utils import dropout_adj, contains_isolated_nodes
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import random 

from models.intrinsic_dimension import computeLID, computeID
from models.utils import *
from models.first_eval_protocol_func import *
from models.second_eval_protocol_func import *

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 labels, edges: torch.Tensor, tau: float = 0.5, walk_length: int = 5,
                 p: float = 1, q: float = 1, disk: float = 0.5):
        super(Model, self).__init__()
        
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.edges = edges
        self.walk_length: int = walk_length
        self.p: float = p
        self.q: float = q
        self.labels = labels
        self.disk: float = disk

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return torch.mm(z1, z2.t())
    
    def keepsame(self, vector, vl):
        return [i for j,i in enumerate(vector) if vl[j] < self.disk]
   
    def knn(self, z: torch.Tensor) -> torch.Tensor:
        z = z.detach().cpu().numpy()
        ed = euclidean_distances(z)
        vl = np.sort(ed)[:,:5]
        sr = np.argsort(ed)[:,:5]

        p = torch.Tensor(np.array([z[self.keepsame(i, vl[j])].mean(0) for j,i in enumerate(sr)]))
        return p

    def proximity(self, z: torch.Tensor) -> torch.Tensor:
        
        return self.knn(z)


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        device = z1.device
        f = lambda x1,x2: torch.exp(torch.mm(x1,x2)/ self.tau)
        intra_view = f(z1, self.proximity(z1).t().to(device))
        inter_view = f(z1, self.proximity(z2).t().to(device))
        neg_aug = f(z1, self.proximity(z3).t().to(device))
        
        res = inter_view.diag() / (intra_view.sum(1) + inter_view.sum(1) 
        + neg_aug.sum(1) - intra_view.diag() - inter_view.diag()- neg_aug.diag())
        
        return -torch.log(res)


    def loss(self, z1: torch.Tensor, z2: torch.Tensor, 
             z3: torch.Tensor, mean: bool = True):
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        z3 = self.projection(z3)
        
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        z3 = F.normalize(z3)

        l1 = self.semi_loss(z1, z2, z3)
        l2 = self.semi_loss(z2, z1, z3)
      

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def train_and_extract_ID_LID_proximity_level(  
            data,
            pretrained_weights = None,
            pretrained_model = None,
            pretrained = False,
            learning_rate = 0.0005, 
            num_hidden = 128, 
            num_proj_hidden = 128,
            activation = nn.PReLU(),
            base_model = GCNConv,
            num_layers = 2,
            drop_edge_rate_1 = 0.2,
            drop_edge_rate_2 = 0.4,
            drop_feature_rate_1 = 0.3,
            drop_feature_rate_2 = 0.4,
            drop_scheme = 'uniform',
            tau = 0.4,
            walk_length = 5,
            p = 1,
            q = 1,
            disk = 0.45,
            num_epochs = 200,
            num_epochs_eval = 500,
            num_classes = 7,
            weight_decay = 0.00001,
            rd_seed = 129,
        ):
          
    torch.manual_seed(rd_seed)
    random.seed(rd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    data = data.to(device)


    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    
    if data.has_isolated_nodes():
        all = torch.arange(start=0, end=data.x.shape[0], step=1, dtype=int).tolist()
        start = torch.unique(data.edge_index[0]).tolist()
        isolated = [i for i in all if i not in start]
        isolated_tensor = torch.Tensor([isolated, isolated])
        all_nodes = torch.cat((data.edge_index, isolated_tensor), dim = 1).type(torch.int64)
        all_nodes_sorted = all_nodes[:, all_nodes[0, :].sort()[1]]
        model = Model(encoder, num_hidden, num_proj_hidden, data.y, all_nodes_sorted, tau, walk_length, p, q, disk).to(device)
    else: 
        model = Model(encoder, num_hidden, num_proj_hidden, data.y, data.edge_index, tau, walk_length, p, q, disk).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if pretrained: 
        model.load_state_dict(torch.load(pretrained_weights))   
    
    dict = {'ID_global_mean' : [],
            'LID_mean' : [],
            'accuracy' : [],
            'kmeans': [],}
    
    for epoch in range(1, num_epochs + 1):
        
        optimizer.zero_grad()

        edge_index_1 = drop_edge(data, device, drop_scheme, drop_edge_rate_1)
        edge_index_2 = drop_edge(data, device, drop_scheme, drop_edge_rate_2)
        
        x_1 = drop_feature_global(data, device, drop_scheme, drop_feature_rate_1)
        x_2 = drop_feature_global(data, device, drop_scheme, drop_feature_rate_2)
        x_3 = shuffle(data.x)
        
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        z3 = model(x_3, data.edge_index)

        loss = model.loss(z1, z2, z3)
        loss.backward()
        optimizer.step()

        print(f'(T) | Epoch={epoch:03d}, loss={loss.item():.4f}')

        res = classifier(model, data, LogReg(num_hidden, num_classes),device=device, n_epochs = num_epochs_eval)
        dict['accuracy'].append(res[0])
        dict['kmeans'].append(clustering_evaluation(model, data, num_classes))

        ID = []
        LID = []

        z = model(data.x, data.edge_index)

        for k in range(num_classes):
            ID.append(computeID(z[data.y == k]))
            LID.append(computeLID(z[data.y == k]))
                
        ID = np.asarray(ID)
        LID = np.asarray(LID)
        ID_global_mean = np.mean(ID)
        LID_mean = np.mean(LID)
        dict['LID_mean'].append(LID_mean)
        dict['ID_global_mean'].append(ID_global_mean)     
    
    print("=== Final ===")
    # torch.save(model, 'node_prox.pth')
    # torch.save(model.state_dict(), 'node_prox_para.pth')
        
    return dict