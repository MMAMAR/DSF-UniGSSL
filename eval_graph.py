import argparse, yaml
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, Amazon
from yaml import SafeLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import models.second_eval_func as evals
from models.logreg import LogReg

from models.graph_level import train_and_extract_ID_LID_graph_level
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(32)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora")
parser.add_argument("--pretrained", type=bool, default=False) # True for finetuning, False otherwise
parser.add_argument("--pretrained_type", type=str, default="none") # cluster, graph, node, prox
parser.add_argument("--device_number", type=int, default=2)
parser.add_argument("--config", type=str, default="config/config-graph-nodecls.yml")
parser.add_argument("--id_lid_file", type=str, default="./nodecls_outputs/pretraining_outputs/graph/cora_LID_ID.csv")
args = parser.parse_args()

if not args.pretrained:
    model_save_folder = "pretrained_models/{}/full_model/graph.pth".format(args.dataset)
    weights_save_folder = "pretrained_models/{}/weights/graph.pth".format(args.dataset)

    pretrained_model_folder = ""
    pretrained_weights_folder = ""

else:
    model_save_folder = "finetuned_models/{}/full_model/{}_graph.pth".format(args.dataset, args.pretrained_type)
    weights_save_folder = "finetuned_models/{}/weights/{}_graph.pth".format(args.dataset, args.pretrained_type)

    pretrained_model_folder = 'pretrained_models/{}/full_model/{}.pth'.format(args.dataset, args.pretrained_type)
    pretrained_weights_folder = 'pretrained_models/{}/weights/{}.pth'.format(args.dataset, args.pretrained_type)

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
config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
device = torch.device("cuda:{}".format(args.device_number))

gg_model, gg_res = train_and_extract_ID_LID_graph_level(data = dataset[0], 
                                                        save = True,
                                                        model_path = model_save_folder,
                                                        weight_path = weights_save_folder,
                                                        pretrained=args.pretrained, 
                                                        pretrained_model=pretrained_model_folder,
                                                        pretrained_weights=pretrained_weights_folder,
                                                        plot = False, 
                                                        learning_rate=config["learning_rate"],
                                                        num_hidden = config["num_hidden"], 
                                                        num_proj_hidden = config["num_proj_hidden"], 
                                                        drop_edge_rate_1 = config["drop_edge_rate_1"], 
                                                        drop_edge_rate_2 = config["drop_edge_rate_2"], 
                                                        drop_feature_rate_1 = config["drop_feature_rate_1"], 
                                                        drop_feature_rate_2 = config["drop_feature_rate_2"], 
                                                        drop_scheme = config["drop_scheme"],
                                                        tau=config["tau"],
                                                        num_epochs = config["num_epochs"],
                                                        patience=config["patience"],
                                                        num_epochs_eval = 500, 
                                                        num_classes = num_classes, 
                                                        weight_decay = config["weight_decay"], 
                                                        rd_seed = 129,
                                                        device=device,
                                                        id_lid_file=args.id_lid_file)
pd.DataFrame(gg_res).accuracy.plot()
print(pd.DataFrame(gg_res).max())
evals.classifier(gg_model, dataset, LogReg(config["num_hidden"], num_classes), device, n_epochs = 500)
