import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import numpy as np
import random

from models.intrinsic_dimension import computeLID, computeID

from models.utils import *
from models.first_eval_protocol_func import *

from tqdm import tqdm

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class Model(nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5, num_clust: int = 7, update = 10, alpha: float = 1):
        super(Model, self).__init__()
        
        self.encoder: Encoder = encoder
        self.tau = tau
        self.cluster_number = num_clust

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        
        self.embedding_dimension = num_hidden
        self.alpha = alpha
        self.update = update

        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.p1 = None
        self.p2 = None

    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x, edge_index)
        return enc
    
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def clustering(self, embeddings, num_clust):
        embeddings = embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=num_clust).fit(embeddings)
        centroids = kmeans.cluster_centers_ 
        preds = kmeans.predict(embeddings)
        preds = np.eye(self.cluster_number)[preds]
        preds = torch.tensor(preds, dtype=torch.float)
        centroids = torch.tensor(centroids, dtype=torch.float)
        return centroids, preds

    
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, counter,
             mean: bool = True):
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        z3 = self.projection(z3)
        
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        z3 = F.normalize(z3)

        if (counter==1) or (counter%self.update==0):
            c1, p1 = self.clustering(z1, self.cluster_number)
            c2, p2 = self.clustering(z2, self.cluster_number)
            c3, _ = self.clustering(z3, self.cluster_number)
            
            self.c1, self.c2, self.c3, self.p1, self.p2 = c1, c2, c3, p1, p2
        else:
            c1, c2, c3, p1, p2 = self.c1, self.c2, self.c3, self.p1, self.p2

        device = z1.device
        c1, c2, c3 = c1.to(device), c2.to(device), c3.to(device)
        p1, p2 = p1.to(device), p2.to(device)


        f = lambda x1, x2, x3 : torch.exp(torch.mul(torch.mm(x1, x2), x3)/ self.tau)

        l1 =  -torch.log(f(z1,c2.T,p1).sum(1) / (f(z1,c1.T, (1-p1)).sum(1) + f(z1,c2.T, (1-p1)).sum(1) + f(z1,c3.T, (1-p1)).sum(1)))      
        l2 =  -torch.log(f(z2,c1.T,p2).sum(1) / (f(z2,c2.T, (1-p2)).sum(1) + f(z2,c1.T, (1-p2)).sum(1) + f(z2,c3.T, (1-p2)).sum(1)))

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

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


def train_and_extract_ID_LID_cluster_level(  
            data,
            save = False,
            model_path = 'pretrained_models/cora/full_model/cluster.pth',
            weight_path = 'pretrained_models/cora/weights/cluster.pth',
            pretrained_weights = None,
            pretrained_model = None,
            pretrained = False,
            plot = True, 
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
            update = 10, 
            rd_seed = 129,
            device=torch.device("cpu"),
            id_lid_file="",
        ):
          
    torch.manual_seed(rd_seed)
    random.seed(rd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = data.to(device)


    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau, num_classes, update).to(device)
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
        for epoch in range(1, num_epochs+1):
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


            loss = model.loss(z1, z2, z3, epoch)
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