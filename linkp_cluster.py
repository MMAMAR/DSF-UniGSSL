import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 
import numpy as np
import random
import math

from models.utils import *
from models.first_eval_protocol_func import *

from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score

import os, datetime
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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
    

def link_decoder(h, edge):
    src_x = h[edge[0]]
    dst_x = h[edge[1]]
    x = (src_x * dst_x).sum(1)
    return x

def do_edge_split_direct(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset.clone()
    num_nodes = data.num_nodes
    row, col = data.edge_index
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
    neg_edge_index = negative_sampling(data.edge_index, num_nodes=num_nodes, num_neg_samples=row.size(0))
    data.val_neg_edge_index = neg_edge_index[:, :n_v]
    data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
    data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true):
    train_auc = roc_auc_score(train_true, train_pred)
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)

    return {'auc_train':train_auc, 'auc_val': valid_auc, 'auc_test': test_auc}

@torch.no_grad()
def test_link_prediction(model, predictor, data, split_edge, batch_size):
    model.eval()
    h = model(data.x, data.edge_index)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['test']['edge'].to(data.x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(data.x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h, edge).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h, edge).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(h, edge).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_pred = torch.cat([pos_train_pred, neg_train_pred], dim=0)
    train_true = torch.cat([torch.ones_like(pos_train_pred), torch.zeros_like(neg_train_pred)], dim=0)

    val_pred = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    val_true = torch.cat([torch.ones_like(pos_valid_pred), torch.zeros_like(neg_valid_pred)], dim=0)

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)

    results = evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true)

    return results


from torch_geometric.datasets import Planetoid, CitationFull, Amazon
from tqdm import tqdm
import argparse, yaml
from yaml import SafeLoader

import warnings
warnings.filterwarnings('ignore')

rd_seed = 412 
torch.manual_seed(rd_seed)
random.seed(rd_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora") # cluster, graph, node, prox
parser.add_argument("--pretrained", type=bool, default=False) # True for finetuning, False otherwise
parser.add_argument("--pretrained_type", type=str, default="none") # cluster, graph, node, prox
parser.add_argument("--epochs", type=int, default=4000)
parser.add_argument("--patience", type=int, default=500)
parser.add_argument("--drop_scheme", type=str, default="uniform") # degree, uniform
parser.add_argument("--device_number", type=int, default=2)
parser.add_argument("--config", type=str, default="config/config-cluster-linkp.yml")
parser.add_argument("--nb_runs", type=int, default=2)
args = parser.parse_args()

if not args.pretrained:
    save_path = "pretrained_models/{}/linkp/cluster.pth".format(args.dataset)
    pretrained_path = "{}"

else:
    save_path = "finetuned_models/{}/linkp/{}_cluster.pth".format(args.dataset, args.pretrained_type)
    pretrained_path = "pretrained_models/{}/linkp/{}.pth".format(args.dataset, args.pretrained_type)

if args.dataset == "cora":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

elif args.dataset == "citeseer":
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    
elif args.dataset == "pubmed":
    dataset = Planetoid(root='/tmp/pubmed', name='pubmed')

elif args.dataset == "dblp":
    dataset = CitationFull(root='/tmp/DBLP', name='dblp')

elif args.dataset == "photo":
    dataset = Amazon(root='/tmp/photo', name='photo')

elif args.dataset == "computers":
    dataset = Amazon(root='/tmp/computers', name='computers')

else:
    print("Unknown dataset:", args.dataset)
    exit()

data = dataset[0]
config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

device = torch.device("cuda:{}".format(args.device_number))
data = data.to(device)
split_edge = do_edge_split_direct(data)
data.edge_index = to_undirected(split_edge['train']['edge'].t())


learning_rate = config["learning_rate"]#0.0005
num_hidden = config["num_hidden"]
num_proj_hidden = config["num_proj_hidden"]
activation = nn.PReLU()
base_model = GCNConv
num_layers = 2
drop_edge_rate_1 = config["drop_edge_rate_1"]#0.2
drop_edge_rate_2 = config["drop_edge_rate_2"]#0.4
drop_feature_rate_1 = config["drop_feature_rate_1"]#0.3
drop_feature_rate_2 = config["drop_feature_rate_2"]#0.4
drop_scheme = args.drop_scheme
tau = config["tau"]#0.4
weight_decay = config["weight_decay"]#0.00001
num_epochs_eval = 500
update = 10

auc_scores = []  
for _ in range(args.nb_runs):
    encoder = Encoder(data.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau, dataset.num_classes, update).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if args.pretrained: 
        model.load_state_dict(torch.load(pretrained_path)) 

    predictor = link_decoder

    best_valid = 0.0
    best_epoch = 0
    cnt_wait = 0
    best_result=0
                
    epochs = args.epochs
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1, epochs + 1):
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

            pbar.set_postfix({'loss': loss.item()})
            pbar.update()
            result = test_link_prediction(model, predictor, data, split_edge, num_hidden)
            valid_hits = result['auc_val']
            if valid_hits > best_valid:
                best_valid = valid_hits
                best_epoch = epoch
                best_result = result
                cnt_wait = 0

                torch.save(model.state_dict(), save_path)

            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                #print('Early stopping!')
                break

        test_auc = best_result['auc_test']
        auc_scores.append(test_auc)

if not args.pretrained:
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {args.dataset}, SingleTask-cluster, Epoch:{best_epoch}, AUC-mean: {np.mean(auc_scores) * 100}, AUC-std: {np.std(auc_scores) * 100}' )
else:
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {args.dataset}, DualStage-({args.pretrained_type}, cluster), Epoch:{best_epoch}, AUC-mean: {np.mean(auc_scores) * 100}, AUC-std: {np.std(auc_scores) * 100}' )

