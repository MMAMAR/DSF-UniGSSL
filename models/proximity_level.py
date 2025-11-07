import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_cluster import random_walk

from torch_geometric.utils import dropout_adj, contains_isolated_nodes
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import matplotlib.pyplot as plt 

from models.utils import *
from models.intrinsic_dimension import computeLID, computeID
from models.first_eval_protocol_func import *

import random, time
from tqdm import tqdm

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 edges: torch.Tensor, tau: float = 0.5, walk_length: int = 5,
                 p: float = 1, q: float = 1):
        super(Model, self).__init__()
        
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.edges = edges.cpu()
        self.walk_length: int = walk_length
        self.p: float = p
        self.q: float = q

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return torch.mm(z1, z2.t())
    
    #def random_walks_mean(self, z: torch.Tensor) -> torch.Tensor:
    #    row, col = self.edges
    #    start = torch.unique(row).to()
    #    walk = random_walk(row=row, col=col, start=start, walk_length=self.walk_length, p=self.p, q=self.q)
    #    walk = walk[:,1:]
    #    walk = walk.detach().cpu().numpy()
    #    znp = z.detach().cpu().numpy()
    #    print(walk)
    #    pr = torch.Tensor([znp[i].mean(0) for i in walk])
    #    return pr
    
    def random_walks_mean(self, z: torch.Tensor) -> torch.Tensor:
        row, col = self.edges
        start = torch.unique(row)

        walk = random_walk(row=row, col=col, start=start, walk_length=self.walk_length, p=self.p, q=self.q)

        # Removing the first step as it's the start node itself
        walk = walk[:, 1:]

        # Handling potential invalid indices (-1 indicates no further node to visit)
        walk_mask = (walk >= 0) & (walk < z.size(0))
        
        # Filter out invalid indices
        filtered_walk = walk * walk_mask - 1 * (~walk_mask)

        # Only calculate mean for valid walks
        pr = z[filtered_walk].mean(dim=1)

        # Only calculate mean for valid walks
        #pr = torch.stack([z[filtered_walk[i, filtered_walk[i] >= 0]].mean(0) if (filtered_walk[i] >= 0).any() else torch.zeros(z.size(1)) for i in range(filtered_walk.shape[0])])
        return pr
    

    def proximity(self, z: torch.Tensor) -> torch.Tensor:    
        return self.random_walks_mean(z)


    # def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        
    #     f = lambda x1,x2: torch.exp(torch.mm(x1,x2)/ self.tau)
    #     device = z1.device

    #     intra_view = f(z1, self.proximity(z1).t().to(device))
    #     inter_view = f(z1, self.proximity(z2).t().to(device))
    #     neg_aug = f(z1, self.proximity(z3).t().to(device))
        
    #     res = inter_view.diag() / (intra_view.sum(1) + inter_view.sum(1) 
    #     + neg_aug.sum(1) - intra_view.diag() - inter_view.diag()- neg_aug.diag())
        
    #     return -torch.log(res)

    # graph_prox: remove intra_view
    # other: remove neg_aug
    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        batch_size = 128
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1

        f = lambda x1,x2: torch.exp(torch.mm(x1,x2)/ self.tau)

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        #p1 = self.proximity(z1).t().to(device)
        p2 = self.proximity(z2).t().to(device)
        p3 = self.proximity(z3).t().to(device)

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            
            #intra_view = f(z1[mask], p1)
            inter_view = f(z1[mask], p2)
            neg_aug = f(z1[mask], p3)

            #losses.append(-torch.log(
            #    inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
            #    / (intra_view.sum(1) + inter_view.sum(1) + neg_aug.sum(1)
            #    - intra_view[:, i * batch_size:(i + 1) * batch_size].diag() 
            #    - inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
            #    - neg_aug[:, i * batch_size:(i + 1) * batch_size].diag())))
            
            losses.append(-torch.log(
                inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
                / (inter_view.sum(1) + neg_aug.sum(1)
                - inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
                - neg_aug[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses) 


    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, mean: bool = True):
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

def visualize(h, color, title):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=45, c=color, cmap="plasma")
    plt.title(title, fontsize = 19, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def train_and_extract_ID_LID_proximity_level(  
            data,
            save = False,
            model_path = 'pretrained_models/cora/full_model/proximity.pth',
            weight_path = 'pretrained_models/cora/weights/proximity.pth',
            plot = True,
            title = None, 
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
            num_epochs = 200,
            patience = 200,
            num_epochs_eval = 500,
            num_classes = 7,
            weight_decay = 0.00001,
            rd_seed = 129,
            kmeans_batch_size=0,
            device=torch.device("cpu"),
            id_lid_file="",
        ):
          
    torch.manual_seed(rd_seed)
    random.seed(rd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = data.to(device)


    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    
    if data.has_isolated_nodes():
        all = torch.arange(start=0, end=data.x.shape[0], step=1, dtype=int).tolist()
        start = torch.unique(data.edge_index[0]).tolist()
        isolated = [i for i in all if i not in start]
        isolated_tensor = torch.Tensor([isolated, isolated]).to(device)
        all_nodes = torch.cat((data.edge_index, isolated_tensor), dim = 1).type(torch.int64)
        all_nodes_sorted = all_nodes[:, all_nodes[0, :].sort()[1]]
        model = Model(encoder, num_hidden, num_proj_hidden, all_nodes_sorted, tau, walk_length, p, q).to(device)
    else: 
        model = Model(encoder, num_hidden, num_proj_hidden, data.edge_index, tau, walk_length, p, q).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if pretrained: 
        model.load_state_dict(torch.load(pretrained_weights)) 
    
    dict = {'accuracy' : [],
            'kmeans': [],
            'nmi': [],
            'ari': []}
    
    best_accuracy = 0
    cpt_wait = 0

    with tqdm(total=num_epochs, desc='(T)') as pbar:
        for epoch in range(1, num_epochs + 1):
            model.train()
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
            
            if (epoch % 1 == 0):
                model.eval()
                with torch.no_grad():
                    accuracy = classifier(model, data, LogReg(num_hidden, num_classes), device, n_epochs = num_epochs_eval)
                    #print(f'(T) | Epoch={epoch:03d}, loss={loss.item():.4f}, accuracy={accuracy:.4f}')
                    dict['accuracy'].append(accuracy)
                    
                    res_clust = clustering_evaluation(model, data, num_classes, batch_size=kmeans_batch_size)
                    dict['kmeans'].append(res_clust[0])
                    dict['nmi'].append(res_clust[1])
                    dict['ari'].append(res_clust[2])

                    pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy, "nmi": res_clust[1]})
                    ID = []
                    LID = []
                    z = model(data.x, data.edge_index)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                    if save:
                        torch.save(model, model_path)
                        torch.save(model.state_dict(), weight_path)

                    else:
                        cpt_wait += 1
                        if cpt_wait > patience:
                            break 
                """
                for k in range(num_classes):
                    ID.append(computeID(z[data.y == k]))
                    LID.append(computeLID(z[data.y == k]))
                
                ID = np.asarray(ID)
                LID = np.asarray(LID)
                ID_global_mean = np.mean(ID)
                LID_mean = np.mean(LID)
                dict['LID_mean'].append(LID_mean)
                dict['ID_global_mean'].append(ID_global_mean)   
                """
            pbar.update()
    
    print("=== Final ===")

    #if save:
    #    torch.save(model, model_path)
    #    torch.save(model.state_dict(), weight_path)

    if plot:
        visualize(z, data.y, title)

    model.load_state_dict(torch.load(weight_path))
    """
    LID_list = [float(el) for el in dict['LID_mean']]
    ID_list = [float(el) for el in dict['ID_global_mean']]
    df = pd.DataFrame({
    "LID": LID_list,
    "ID": ID_list
    })
    df.to_csv(id_lid_file, index=False)
    """
    return model, dict