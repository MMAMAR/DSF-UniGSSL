import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cluster import random_walk

from torch.utils.data import DataLoader

from torch_geometric.nn import GCNConv

from torch_geometric.utils import contains_isolated_nodes, negative_sampling, to_undirected

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import matplotlib.pyplot as plt 

from models.utils import *
from sklearn.metrics import roc_auc_score, average_precision_score
import random 
import math, datetime


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
    
    def random_walks_mean(self, z: torch.Tensor) -> torch.Tensor:
        row, col = self.edges
        start = torch.unique(row)
        walk = random_walk(row=row, col=col, start=start, walk_length=self.walk_length, p=self.p, q=self.q)
        walk = walk[:,1:]
        walk = walk.detach().cpu().numpy()
        znp = z.detach().cpu().numpy()
        pr = torch.Tensor([znp[i].mean(0) for i in walk])
        return pr
    

    def proximity(self, z: torch.Tensor) -> torch.Tensor:
        
        return self.random_walks_mean(z)


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor) -> torch.Tensor:
        batch_size = 128
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1

        f = lambda x1,x2: torch.exp(torch.mm(x1,x2)/ self.tau)

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        p1 = self.proximity(z1).t().to(device)
        p2 = self.proximity(z2).t().to(device)
        p3 = self.proximity(z3).t().to(device)

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            
            intra_view = f(z1[mask], p1)
            inter_view = f(z1[mask], p2)
            neg_aug = f(z1[mask], p3)

            losses.append(-torch.log(
                inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
                / (intra_view.sum(1) + inter_view.sum(1) + neg_aug.sum(1)
                - intra_view[:, i * batch_size:(i + 1) * batch_size].diag() 
                - inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
                - neg_aug[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses) 



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

rd_seed = 129 
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
parser.add_argument("--device_number", type=int, default=4)
parser.add_argument("--config", type=str, default="config/config-prox-linkp.yml") 
parser.add_argument("--nb_runs", type=int, default=2)
args = parser.parse_args()

if not args.pretrained:
    save_path = "pretrained_models/{}/linkp/prox.pth".format(args.dataset)
    pretrained_path = "{}"

else:
    save_path = "finetuned_models/{}/linkp/{}_prox.pth".format(args.dataset, args.pretrained_type)
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
walk_length = 3
p = 1
q = 1
weight_decay = config["weight_decay"]#0.00001


auc_scores = []  
for _ in range(args.nb_runs):
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

            loss = model.loss(z1, z2, z3)
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
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {args.dataset}, SingleTask-prox, Epoch:{best_epoch}, AUC-mean: {np.mean(auc_scores) * 100}, AUC-std: {np.std(auc_scores) * 100}' )
else:
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {args.dataset}, DualStage-({args.pretrained_type}, prox), Epoch:{best_epoch}, AUC-mean: {np.mean(auc_scores) * 100}, AUC-std: {np.std(auc_scores) * 100}' )
