import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from torch_geometric.utils import dropout_adj
import numpy as np
import pandas as pd
from models.intrinsic_dimension import computeLID, computeID
from models.first_eval_protocol_func import *
from models.utils import *
from tqdm import tqdm

import random 

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class Model(nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        
        self.encoder: Encoder = encoder
        self.tau = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x, edge_index)
        return enc
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def readout(self, z: torch.Tensor) -> torch.Tensor:
        s = z.mean(dim = 0)
        return torch.unsqueeze(s, dim=1)
    
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
             mean: bool = True):
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        z3 = self.projection(z3)
        
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        z3 = F.normalize(z3)

        s1 = self.readout(z1)
        s2 = self.readout(z2)
        s3 = self.readout(z3)

        ps = lambda x1,x2: torch.exp(torch.mm(x1,x2)/ self.tau)

        l1 =  -torch.log(ps(z1, s2) / ps(z1, s3))
        l2 =  -torch.log(ps(z2, s1) / ps(z2, s3))

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def train_graph_level(  
            data,
            model_path = 'pretrained_models/cora/full_model/graph.pth',
            weight_path = 'pretrained_models/cora/weights/graph.pth',
            save = True,
            learning_rate = 0.001, 
            num_hidden = 128, 
            num_proj_hidden = 128,
            activation = F.relu,
            base_model = GCNConv,
            num_layers = 2,
            drop_edge_rate_1 = 0.2,
            drop_edge_rate_2 = 0.4,
            drop_feature_rate_1 = 0.3,
            drop_feature_rate_2 = 0.4,
            drop_scheme = 'uniform',
            tau = 0.4,
            num_epochs = 200,
            weight_decay = 0.00001
        ):
          
    torch.manual_seed(39788)
    random.seed(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)


    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    print("=== Final ===")

    if save:
        torch.save(model, model_path)
        torch.save(model.state_dict(), weight_path)
    
    return model



def visualize(h, color, title):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu())
    color = color.detach().cpu()
    
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=45, c=color, cmap="plasma")
    plt.title(title, fontsize = 19, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def train_and_extract_ID_LID_graph_level(  
            data,
            save = False,
            model_path = 'pretrained_models/cora/full_model/graph.pth',
            weight_path = 'pretrained_models/cora/weights/graph.pth',
            pretrained_weights = None,
            pretrained_model = None,
            pretrained = False,
            plot = False, 
            title = None,
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
            num_epochs = 200,
            patience=50,
            num_epochs_eval = 500,
            num_classes = 7,
            weight_decay = 0.00001,
            rd_seed = 129,
            device=torch.device("cpu"),
            id_lid_file=""
        ):
          
    torch.manual_seed(rd_seed)
    random.seed(rd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = data.to(device)


    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
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

                    res_clust = clustering_evaluation(model, data, num_classes)
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