import argparse, yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, Amazon
from yaml import SafeLoader
from torch_geometric.nn import GCNConv

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import models.second_eval_func as evals
from models.logreg import LogReg
import random
from models.filter_level import train_filter_level


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora")
parser.add_argument("--pretraining_type", type=str, default="none")  # cluster, graph, node, prox
parser.add_argument("--finetuning_type", type=str, default="none")  # cluster, graph, node, prox
parser.add_argument("--device_number", type=int, default=2)
parser.add_argument("--filtering_threshold", type=float, default=1.0)
parser.add_argument("--pca_n_comp", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--cls_epochs", type=int, default=100)
parser.add_argument("--id_lid_file", type=str, default="./nodecls_outputs/filter_outputs/cora/node_prox_LID_ID.csv")
args = parser.parse_args()
set_seed(args.seed)
config_pretraining = "config/config-{}-nodecls.yml".format(args.pretraining_type)
config_finetuning = "config/config-{}-nodecls.yml".format(args.finetuning_type)

model_save_folder = "filter_models/{}/full_model/{}_{}.pth".format(args.dataset, args.pretraining_type,
                                                                   args.finetuning_type)
weights_save_folder = "filter_models/{}/weights/{}_{}.pth".format(args.dataset, args.pretraining_type,
                                                                  args.finetuning_type)

pretrained_model_folder = 'pretrained_models/{}/full_model/{}.pth'.format(args.dataset, args.pretraining_type)
pretrained_weights_folder = 'pretrained_models/{}/weights/{}.pth'.format(args.dataset, args.pretraining_type)

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

num_classes = dataset.num_classes
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

device = torch.device("cuda:{}".format(args.device_number))

cc_model, cc_res = train_filter_level(
    dataset[0],
    save=True,
    pretraining_type=args.pretraining_type,
    finetuning_type=args.finetuning_type,
    save_model_path=model_save_folder,
    save_weight_path=weights_save_folder,
    pretrained_path=pretrained_weights_folder,
    plot=False,
    title=None,
    pretraining_args=pretraining_args,
    finetuning_args=finetuning_args,
    device=device,
    id_lid_file=args.id_lid_file,
    filtering_threshold=args.filtering_threshold,
    pca_n_comp=args.pca_n_comp,
)

print('nmi: ',pd.DataFrame(cc_res).max()['nmi'].round(4))
print('ari: ', pd.DataFrame(cc_res).max()['ari'].round(4))