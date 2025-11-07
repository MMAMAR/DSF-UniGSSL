############################ EVAL ############################

import torch 
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from models.logreg import LogReg
import functools
from munkres import Munkres
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


torch.manual_seed(19)
torch.use_deterministic_algorithms(True)


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean'] * 100
        std = statistics[key]['std'] * 100
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

def train_lrc(X_train, y_train, model, optimizer, criterion, device):
    model.to(device).train()
    optimizer.zero_grad()  
    out = model(X_train)  
    loss = criterion(out, y_train)  
    loss.backward()  
    optimizer.step()  
    return loss

def test_lrc(X_test, y_test, model, device):
    model.to(device).eval()
    out = model(X_test)
    pred = out.argmax(dim=1)  
    return accuracy_score(pred.detach().cpu().numpy(), y_test.detach().cpu().numpy())


@repeat(5)
def classifier(ssl_model, data, clf_model, device, n_epochs = 500):
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=0.01, weight_decay=5e-4)  
    embedding = ssl_model(data.x.to(device), data.edge_index.to(device))

    X = embedding.detach()
    Y = data.y.detach()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9)
    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
    
    for epoch in range(0, n_epochs):
        loss = train_lrc(X_train, y_train, clf_model, optimizer, criterion, device)
    acc = test_lrc(X_test, y_test, clf_model, device)
    
    return {
        'acc': acc,
    }


def get_matches(y_true, y_pred):
    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)
       

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]

            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    return new_predict


@repeat(5)
def clustering_evaluation(model, data, num_clusters):
    model.eval()
    z = model(data.x, data.edge_index)
    X = z.detach().cpu().numpy()
    Y = data.y.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    pred_label1 = kmeans.fit_predict(X)

    new_predict1 = get_matches(Y, pred_label1)

    acc = accuracy_score(Y, new_predict1)
    
    return {
        'acc': acc,
    } 


def evaluate_auc(train_pred, train_true, val_pred, val_true, test_pred, test_true):
    train_auc = roc_auc_score(train_true, train_pred)
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    train_ap = average_precision_score(train_true, train_pred)
    valid_ap = average_precision_score(val_true, val_pred)
    test_ap = average_precision_score(test_true, test_pred)
    results = dict()
    results['AUC'] = (train_auc, valid_auc, test_auc)
    results['AP'] = (train_ap, valid_ap, test_ap)
    return results

@torch.no_grad()
def test_link_prediction(model, predictor, data, split_edge, batch_size):
    model.eval()
    h = model.encoder(data.x, data.edge_index)

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
