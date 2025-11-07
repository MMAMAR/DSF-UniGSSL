import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_cluster import random_walk

import numpy as np
import random

from models.intrinsic_dimension import computeLID, computeID

from models.utils import *
from models.first_eval_protocol_func import *
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from torch_pca import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch.utils.data import DataLoader

import numpy as np
import random
import math

from models.utils import *

from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import Planetoid, CitationFull, Amazon
from tqdm import tqdm
import argparse, yaml
from yaml import SafeLoader

import warnings
import os, datetime

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

model = None
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

def projection(z: torch.Tensor) -> torch.Tensor:
    z = F.elu(model.fc1(z))
    return model.fc2(z)

def sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return torch.mm(z1, z2.t())

# --------------------------------------------- NODE -------------------------------------------------------------------- #
def node_level_semi_loss(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, args) -> torch.Tensor:

    f = lambda x1, x2: torch.exp(torch.mm(x1,x2) / args["tau"])
    intra_view = f(z1, z1.t())
    inter_view = f(z1, z2.t())
    neg_aug = f(z1, z3.t())

    res = inter_view.diag() / (intra_view.sum(1) + inter_view.sum(1)
    + neg_aug.sum(1) - intra_view.diag() - inter_view.diag() - neg_aug.diag())

    return -torch.log(res)

def node_level_semi_loss_batch(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, args) -> torch.Tensor:
    batch_size = args["batch"]
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1

    f = lambda x1,x2: torch.exp(torch.mm(x1,x2)/ args["tau"])

    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]

        intra_view = f(z1[mask], z1.t())
        inter_view = f(z1[mask], z2.t())
        neg_aug = f(z1[mask], z3.t())

        losses.append(-torch.log(
            inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
            / (intra_view.sum(1) + inter_view.sum(1) + neg_aug.sum(1)
            - intra_view[:, i * batch_size:(i + 1) * batch_size].diag()
            - inter_view[:, i * batch_size:(i + 1) * batch_size].diag()
            - neg_aug[:, i * batch_size:(i + 1) * batch_size].diag() )))

    return torch.cat(losses)

def node_level_loss(z1: torch.Tensor, z2: torch.Tensor,
            z3: torch.Tensor, args, mean: bool = True):
    z1 = projection(z1)
    z2 = projection(z2)
    z3 = projection(z3)

    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    z3 = F.normalize(z3)

    if (args["batch"] == 0):
        l1 = node_level_semi_loss(z1, z2, z3, args)
        l2 = node_level_semi_loss(z2, z1, z3, args)
    else:
        l1 = node_level_semi_loss_batch(z1, z2, z3, args)
        l2 = node_level_semi_loss_batch(z2, z1, z3, args)


    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret

# --------------------------------------------- PROXIMITY -------------------------------------------------------------------- #
def proximity_mean_std(z: torch.Tensor, args, filtering_threshold, pca_n_comp):
    z = z.detach().cpu()
    if (args["counter"] - 1) % 20 == 0:
        row, col = args["edges"]
        start = torch.range(0, z.shape[0] - 1, dtype=torch.long)
        walk = random_walk(row=row, col=col, start=start, walk_length=args["walk_length"], p=args["p"], q=args["q"])

        # Removing the first step as it's the start node itself
        walk = walk[:, 1:]

        # Handling potential invalid indices (-1 indicates no further node to visit)
        walk_mask = (walk >= 0) & (walk < z.size(0))

        # Filter out invalid indices
        filtered_walk = walk * walk_mask - 1 * (~walk_mask)
        args["filtered_walk"] = filtered_walk

    else:
        filtered_walk = args["filtered_walk"]

    # Only calculate mean for valid walks
    pca_model = PCA(n_components=pca_n_comp, svd_solver="auto")
    z_project = pca_model.fit_transform(z)
    

    with torch.no_grad():
        if (args["counter"] - 1) % 1 == 0:
            mean = z_project[filtered_walk].mean(dim=1)
            std = z_project[filtered_walk].std(dim=1)

            args["mean"] = mean
            args["std"] = std
        else:
            mean = args["mean"]
            std = args["std"]

    mask = (torch.abs(z_project - mean) < filtering_threshold * std).all(dim=1).cpu()

    idx = torch.arange(start=0, end=z.shape[0])
    pretraining_idx = idx[~mask]
    finetuning_idx = idx[mask]

    return pretraining_idx, finetuning_idx

def random_walks_mean(z: torch.Tensor, args) -> torch.Tensor:
    z = projection(z)
    z = F.normalize(z)

    row, col = args["edges"]
    start = torch.unique(row)

    walk = random_walk(row=row, col=col, start=start, walk_length=args["walk_length"], p=args["p"], q=args["q"])

    # Removing the first step as it's the start node itself
    walk = walk[:, 1:]

    # Handling potential invalid indices (-1 indicates no further node to visit)
    walk_mask = (walk >= 0) & (walk < z.size(0))

    # Filter out invalid indices
    filtered_walk = walk * walk_mask - 1 * (~walk_mask)

    # Only calculate mean for valid walks
    pr = z[filtered_walk].mean(dim=1)

    return pr


def proximity(z: torch.Tensor, args) -> torch.Tensor:
    return random_walks_mean(z, args)


def proximity_level_semi_loss(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, args) -> torch.Tensor:
    batch_size = 128
    device = z1.device

    f = lambda x1,x2: torch.exp(torch.mm(x1,x2)/ args["tau"])

    losses = []


    p2 = proximity(z2, args).t().to(device)
    p3 = proximity(z3, args).t().to(device)

    num_nodes = min(z1.shape[0], p2.shape[1])
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes).to(device)

    for i in range(num_batches):
        idx_start = i * batch_size
        idx_end = min((i + 1) * batch_size, num_nodes)

        mask = indices[idx_start:idx_end]

        #intra_view = f(z1[mask], p1)
        inter_view = f(z1[mask], p2)
        neg_aug = f(z1[mask], p3)



        losses.append(-torch.log(
            inter_view[:, idx_start:idx_end].diag()
            / (inter_view.sum(1) + neg_aug.sum(1)
            - inter_view[:, idx_start:idx_end].diag()
            - neg_aug[:, idx_start:idx_end].diag())))

    return torch.cat(losses)


def proximity_level_loss(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, args, mean: bool = True):
    z1 = projection(z1)
    z2 = projection(z2)
    z3 = projection(z3)

    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    z3 = F.normalize(z3)

    l1 = proximity_level_semi_loss(z1, z2, z3, args)
    l2 = proximity_level_semi_loss(z2, z1, z3, args)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret

# --------------------------------------------- CLUSTER -------------------------------------------------------------------- #
def cluster_mean_std(z: torch.Tensor, args, filtering_threshold, pca_n_comp):
    z = projection(z)
    z = F.normalize(z)
    z = z.detach().cpu()
    pca_model = PCA(n_components=pca_n_comp, svd_solver="auto")
    z = pca_model.fit_transform(z)

    z = z.numpy()
    if (args["counter"] - 1) % 1 == 0:
        gmm = GaussianMixture(n_components=args['num_classes'], covariance_type="diag").fit(z)
        preds = gmm.predict(z)

        centers = gmm.means_[preds]
        cov = gmm.covariances_[preds]

        args["centers"] = centers
        args["cov"] = cov

    else:
        centers = args["centers"]
        cov = args["cov"]

    mask = (np.abs(z - centers) < filtering_threshold * cov).all(axis=1)
    idx = torch.arange(start=0, end=z.shape[0])
    pretraining_idx = idx[~mask]
    finetuning_idx = idx[mask]

    return pretraining_idx, finetuning_idx

def clustering(embeddings, num_clust):
    embeddings = embeddings.detach().cpu().numpy()
    kmeans = GaussianMixture(n_components=num_clust, covariance_type="diag")
    kmeans = kmeans.fit(embeddings)
    centroids = kmeans.means_
    preds = kmeans.predict(embeddings)
    preds = np.eye(num_clust)[preds]
    preds = torch.tensor(preds, dtype=torch.float)
    centroids = torch.tensor(centroids, dtype=torch.float)
    return centroids, preds


def cluster_level_loss(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, args,
            mean: bool = True):
    z1 = projection(z1)
    z2 = projection(z2)
    z3 = projection(z3)

    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    z3 = F.normalize(z3)

    if (args['counter']==1) or (args['counter']%args['update']==0) or (z1.shape[0] != args['p1'].shape[0]):
        c1, p1 = clustering(z1, args['num_classes'])
        c2, p2 = clustering(z2, args['num_classes'])
        c3, _ = clustering(z3, args['num_classes'])

        args['c1'], args['c2'], args['c3'], args['p1'], args['p2'] = c1, c2, c3, p1, p2
    else:
        c1, c2, c3, p1, p2 = args['c1'], args['c2'], args['c3'], args['p1'], args['p2']

    device = z1.device
    c1, c2, c3 = c1.to(device), c2.to(device), c3.to(device)
    p1, p2 = p1.to(device), p2.to(device)


    f = lambda x1, x2, x3 : torch.exp(torch.mul(torch.mm(x1, x2), x3)/ args['tau'])

    l1 =  -torch.log(f(z1,c2.T,p1).sum(1) / (f(z1,c1.T, (1-p1)).sum(1) + f(z1,c2.T, (1-p1)).sum(1) + f(z1,c3.T, (1-p1)).sum(1)))
    l2 =  -torch.log(f(z2,c1.T,p2).sum(1) / (f(z2,c2.T, (1-p2)).sum(1) + f(z2,c1.T, (1-p2)).sum(1) + f(z2,c3.T, (1-p2)).sum(1)))

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret

# --------------------------------------------- GRAPH -------------------------------------------------------------------- #
def graph_mean_std(z: torch.Tensor, args, filtering_threshold, pca_n_comp):
    z = projection(z)
    z = F.normalize(z)
    z = z.detach().cpu()
    pca_model = PCA(n_components=pca_n_comp, svd_solver="auto")
    z_project = pca_model.fit_transform(z)

    mean = z_project.mean(dim=0)
    std = z_project.std(dim=0)

    mask = (torch.abs(z_project - mean) < filtering_threshold * std).all(dim=1).cpu()

    idx = torch.arange(start=0, end=z.shape[0])
    pretraining_idx = idx[~mask]
    finetuning_idx = idx[mask]

    return pretraining_idx, finetuning_idx

def readout(z: torch.Tensor) -> torch.Tensor:
    s = z.mean(dim = 0)
    return torch.unsqueeze(s, dim=1)

def graph_level_loss(z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, args,
            mean: bool = True):
    z1 = projection(z1)
    z2 = projection(z2)
    z3 = projection(z3)

    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    z3 = F.normalize(z3)

    s1 = readout(z1)
    s2 = readout(z2)
    s3 = readout(z3)

    ps = lambda x1,x2: torch.exp(torch.mm(x1,x2)/ args['tau'])

    l1 =  -torch.log(ps(z1, s2) / ps(z1, s3))
    l2 =  -torch.log(ps(z2, s1) / ps(z2, s3))

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret

# ------------------------------------------------------------------------------------------------------------------------ #
def visualize(h, color, title):
    h = h.detach().cpu()
    z = TSNE(n_components=2).fit_transform(h)
    color = color.detach().cpu()
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=45, c=color, cmap="plasma")
    plt.title(title, fontsize = 19, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('plot.png')
    plt.show()

def train_filter_level(
            data,
            save = False,
            pretraining_type='node',
            finetuning_type='proximity',
            save_model_path = 'pretrained_models/cora/full_model/node.pth',
            save_weight_path = 'pretrained_models/cora/weights/node.pth',
            pretrained_path = None,
            plot = True,
            title = None,
            pretraining_args=None,
            finetuning_args=None,
            device=torch.device("cpu"),
        ):
    global model

    if pretraining_type == 'node':
        loss_pretraining_func = node_level_loss
    elif pretraining_type == 'prox':
        loss_pretraining_func = proximity_level_loss
    elif pretraining_type == 'cluster':
        loss_pretraining_func = cluster_level_loss
    else:
        loss_pretraining_func = graph_level_loss

    if finetuning_type == 'node':
        loss_finetuning_func = node_level_loss
    elif finetuning_type == 'prox':
        loss_finetuning_func = proximity_level_loss
    elif finetuning_type == 'cluster':
        loss_finetuning_func = cluster_level_loss
    else:
        loss_finetuning_func = graph_level_loss

    torch.manual_seed(finetuning_args['seed'])
    random.seed(finetuning_args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data = data.to(device)

    encoder = Encoder(data.num_features, finetuning_args['num_hidden'], finetuning_args['activation'], base_model=finetuning_args['base_model'], k=finetuning_args['num_layers']).to(device)
    model = Model(encoder, finetuning_args['num_hidden'], finetuning_args['num_proj_hidden'], finetuning_args['tau']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetuning_args['learning_rate'], weight_decay=finetuning_args['weight_decay'])

    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)

    dict = {'accuracy' : [],
            'kmeans': [],
            'nmi': [],
            'ari': []}

    best_accuracy = 0
    cpt_wait = 0

    pretraining_args["edges"] = data.edge_index.cpu()
    finetuning_args["edges"] = data.edge_index.cpu()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    with tqdm(total=finetuning_args['num_epochs'], desc='(T)') as pbar:
        for epoch in range(1, finetuning_args['num_epochs'] + 1):
            pretraining_args["counter"] = epoch
            finetuning_args["counter"] = epoch

            model.train()
            optimizer.zero_grad()

            edge_index_1 = drop_edge(data, device, finetuning_args['drop_scheme'], finetuning_args['drop_edge_rate_1'])
            edge_index_2 = drop_edge(data, device, finetuning_args['drop_scheme'], finetuning_args['drop_edge_rate_2'])

            x_1 = drop_feature_global(data, device, finetuning_args['drop_scheme'], finetuning_args['drop_feature_rate_1'])
            x_2 = drop_feature_global(data, device, finetuning_args['drop_scheme'], finetuning_args['drop_feature_rate_2'])
            x_3 = shuffle(data.x)

            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)
            z3 = model(x_3, data.edge_index)

            if finetuning_type == 'prox':
                pretrained_indices, finetuning_indices = proximity_mean_std(z1, finetuning_args)
            elif finetuning_type == 'cluster':
                pretrained_indices, finetuning_indices = cluster_mean_std(z1, finetuning_args)
            elif finetuning_type == "graph":
                pretrained_indices, finetuning_indices = graph_mean_std(z1, finetuning_args)

            print("Nb pretraining:", len(pretrained_indices))
            print("Nb finetuning:", len(finetuning_indices))

            if len(pretrained_indices) > finetuning_args['num_classes']:
                loss_pretraining = loss_pretraining_func(z1[pretrained_indices], z2[pretrained_indices], z3[pretrained_indices], pretraining_args)
            else:
                loss_pretraining = 0

            if len(finetuning_indices) > finetuning_args['num_classes']:
                loss_finetuning = loss_finetuning_func(z1[finetuning_indices], z2[finetuning_indices], z3[finetuning_indices], finetuning_args)
            else:
                loss_finetuning = 0

            loss = loss_pretraining + loss_finetuning
            loss.backward()
            optimizer.step()


            if (epoch % 1 == 0):
                model.eval()
                with torch.no_grad():
                    accuracy = classifier(model, data, LogReg(finetuning_args['num_hidden'], finetuning_args['num_classes']), device, n_epochs = finetuning_args['num_epochs_eval'])
                    #print(f'(T) | Epoch={epoch:03d}, loss={loss.item():.4f}, accuracy={accuracy:.4f}')
                    dict['accuracy'].append(accuracy)

                    res_clust = clustering_evaluation(model, data, finetuning_args['num_classes'])
                    dict['kmeans'].append(res_clust[0])
                    dict['nmi'].append(res_clust[1])
                    dict['ari'].append(res_clust[2])

                    pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy, "nmi": res_clust[1]})

                    #ID = []
                    #LID = []
                    z = model(data.x, data.edge_index)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        if save:
                            torch.save(model, save_model_path)
                            torch.save(model.state_dict(), save_weight_path)

                    else:
                        cpt_wait += 1
                        if cpt_wait > finetuning_args['patience']:
                            break


            pbar.update()

    print("=== Final ===")

    if plot:
        visualize(z, data.y, title)

    model.load_state_dict(torch.load(save_weight_path))
    return model, dict



os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def link_decoder(h, edge):
    src_x = h[edge[0]]
    dst_x = h[edge[1]]
    x = (src_x * dst_x).sum(1)
    return x

def do_edge_split_direct(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.05):
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

    return {'auc_train': train_auc, 'auc_val': valid_auc, 'auc_test': test_auc}


@torch.no_grad()
def test_link_prediction(model, predictor, data, split_edge, batch_size):
    model.eval()
    h = model(data.x, data.edge_index)

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)
    pos_test_edge = split_edge['valid']['edge'].to(data.x.device)
    neg_test_edge = split_edge['valid']['edge_neg'].to(data.x.device)
    pos_valid_edge = split_edge['test']['edge'].to(data.x.device)
    neg_valid_edge = split_edge['test']['edge_neg'].to(data.x.device)

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


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora")  # cluster, graph, node, prox
parser.add_argument("--pretrained", type=bool, default=True)  # True for finetuning, False otherwise
#parser.add_argument("--pretrained_type", type=str, default="none")  # cluster, graph, node, prox
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--patience", type=int, default=1500)
parser.add_argument("--batch_size", type=int, default=0)
parser.add_argument("--drop_scheme", type=str, default="uniform")  # degree, uniform
parser.add_argument("--device_number", type=int, default=0)
parser.add_argument("--config", type=str, default="config/config-node-linkp.yml")
parser.add_argument("--nb_runs", type=int, default=1)
parser.add_argument("--filtering_threshold", type=float, default=2.0)
parser.add_argument("--pca_n_comp", type=int, default=3)
parser.add_argument("--pretraining_type", type=str, default='node')
parser.add_argument("--finetuning_type", type=str, default='prox')
parser.add_argument("--save_model_path", type=str, default='pretrained_models/cora/full_model/node.pth')
parser.add_argument("--save_weight_path", type=str, default='pretrained_models/cora/weights/node.pth')
parser.add_argument("--seed", type=int, default=1)


args = parser.parse_args()
set_seed(args.seed)
if not args.pretrained:
    save_path = "pretrained_models/{}/linkp/node.pth".format(args.dataset)
    pretrained_path = "{}"

else:
    save_path = "finetuned_models/{}/linkp/{}_node.pth".format(args.dataset, args.pretraining_type)
    pretrained_path = "pretrained_models/{}/linkp/{}.pth".format(args.dataset, args.pretraining_type)

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

rd_seed = 129
torch.manual_seed(rd_seed)
random.seed(rd_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:{}".format(args.device_number))
data = data.to(device)
split_edge = do_edge_split_direct(data)
data.edge_index = to_undirected(split_edge['train']['edge'].t())

learning_rate = config["learning_rate"]  # 0.0001
num_hidden = config["num_hidden"]
num_proj_hidden = config["num_proj_hidden"]
activation = nn.PReLU()
base_model = GCNConv
num_layers = 2
drop_edge_rate_1 = config["drop_edge_rate_1"]  # 0.2
drop_edge_rate_2 = config["drop_edge_rate_2"]  # 0.4
drop_feature_rate_1 = config["drop_feature_rate_1"]  # 0.3
drop_feature_rate_2 = config["drop_feature_rate_2"]  # 0.4
drop_scheme = args.drop_scheme
tau = config["tau"]  # 0.4
weight_decay = config["weight_decay"]  # 0.00001
batch = args.batch_size
epochs = args.epochs

if args.pretraining_type == 'node':
    loss_pretraining_func = node_level_loss
elif args.pretraining_type == 'prox':
    loss_pretraining_func = proximity_level_loss
elif args.pretraining_type == 'cluster':
    loss_pretraining_func = cluster_level_loss
else:
    loss_pretraining_func = graph_level_loss

if args.finetuning_type == 'node':
    loss_finetuning_func = node_level_loss
elif args.finetuning_type == 'prox':
    loss_finetuning_func = proximity_level_loss
elif args.finetuning_type == 'cluster':
    loss_finetuning_func = cluster_level_loss
else:
    loss_finetuning_func = graph_level_loss

num_classes = dataset.num_classes
config_pretraining = "config/config-{}-linkp.yml".format(args.pretraining_type)
config_finetuning = "config/config-{}-linkp.yml".format(args.finetuning_type)
pretraining_args = yaml.load(open(config_pretraining), Loader=SafeLoader)[args.dataset]
finetuning_args = yaml.load(open(config_finetuning), Loader=SafeLoader)[args.dataset]

pretraining_args["activation"] = nn.PReLU()
pretraining_args["base_model"] = GCNConv
pretraining_args["num_layers"] = 2
pretraining_args["num_classes"] = num_classes
pretraining_args["num_epochs_eval"] = 500

finetuning_args["activation"] = nn.PReLU()
finetuning_args["base_model"] = GCNConv
finetuning_args["num_layers"] = 2
finetuning_args["num_classes"] = num_classes
finetuning_args["num_epochs_eval"] = 500

torch.manual_seed(finetuning_args['seed'])
random.seed(finetuning_args['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#data = data.to(device)
auc_scores = []
for _ in range(args.nb_runs):
    encoder = Encoder(data.num_features, finetuning_args['num_hidden'], finetuning_args['activation'],
                      base_model=finetuning_args['base_model'], k=finetuning_args['num_layers']).to(device)
    model = Model(encoder, finetuning_args['num_hidden'], finetuning_args['num_proj_hidden'], finetuning_args['tau']).to(
        device)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetuning_args['learning_rate'],
                                 weight_decay=finetuning_args['weight_decay'])

    predictor = link_decoder

    model.load_state_dict(torch.load(pretrained_path,map_location=torch.device("cpu")))
    model.to(device)

    dict = {'accuracy': [],
            'kmeans': [],
            'nmi': [],
            'ari': []}

    best_accuracy = 0
    cpt_wait = 0

    pretraining_args["edges"] = data.edge_index.cpu()
    finetuning_args["edges"] = data.edge_index.cpu()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_valid = 0.0
    with tqdm(total=finetuning_args['num_epochs'], desc='(T)') as pbar:
        for epoch in range(1, finetuning_args['num_epochs'] + 1):
            pretraining_args["counter"] = epoch
            finetuning_args["counter"] = epoch

            model.train()
            optimizer.zero_grad()

            edge_index_1 = drop_edge(data, device, finetuning_args['drop_scheme'], finetuning_args['drop_edge_rate_1'])
            edge_index_2 = drop_edge(data, device, finetuning_args['drop_scheme'], finetuning_args['drop_edge_rate_2'])

            x_1 = drop_feature_global(data, device, finetuning_args['drop_scheme'], finetuning_args['drop_feature_rate_1'])
            x_2 = drop_feature_global(data, device, finetuning_args['drop_scheme'], finetuning_args['drop_feature_rate_2'])
            x_3 = shuffle(data.x)

            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)
            z3 = model(x_3, data.edge_index)

            if args.finetuning_type == 'prox':
                pretrained_indices, finetuning_indices = proximity_mean_std(z1, finetuning_args, args.filtering_threshold, args.pca_n_comp)
            elif args.finetuning_type == 'cluster':
                pretrained_indices, finetuning_indices = cluster_mean_std(z1, finetuning_args, args.filtering_threshold, args.pca_n_comp)
            elif args.finetuning_type == "graph":
                pretrained_indices, finetuning_indices = graph_mean_std(z1, finetuning_args, args.filtering_threshold, args.pca_n_comp)

            if len(pretrained_indices) > finetuning_args['num_classes']:
                loss_pretraining = loss_pretraining_func(z1[pretrained_indices], z2[pretrained_indices],
                                                         z3[pretrained_indices], pretraining_args)
            else:
                loss_pretraining = 0

            if len(finetuning_indices) > finetuning_args['num_classes']:
                loss_finetuning = loss_finetuning_func(z1[finetuning_indices], z2[finetuning_indices],
                                                       z3[finetuning_indices], finetuning_args)
            else:
                loss_finetuning = 0

            loss = loss_pretraining + loss_finetuning
            loss.backward()
            optimizer.step()

            if (epoch % 1 == 0):
                model.eval()
                with torch.no_grad():
                    result = test_link_prediction(model, predictor, data, split_edge, num_hidden)
                    valid_hits = result['auc_val']
                    if valid_hits > best_valid:
                        best_valid = valid_hits
                        best_epoch = epoch
                        best_result = result
                        cnt_wait = 0

                        torch.save(model.state_dict(), save_path)


                    pbar.set_postfix({"loss": loss.item()})

            pbar.update()
        test_auc = best_result['auc_test']
        auc_scores.append(test_auc)

print("=== Final ===")
print(
        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {args.dataset}, DualStage with Filtering-({args.pretraining_type}, {args.finetuning_type} ), AUC-mean: {np.mean(auc_scores) * 100}, AUC-std: {np.std(auc_scores) * 100}')

