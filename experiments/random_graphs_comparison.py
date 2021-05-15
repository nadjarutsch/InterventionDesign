import torch
from argparse import ArgumentParser
from datetime import datetime
import json
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import sys
sys.path.append("../")

from DAG_generation.graph_discovery import GraphDiscoveryDAG
from DAG_pairwise.graph_discovery import GraphDiscoveryPairwise
from DAG_matrix.graph_discovery import GraphDiscoveryMatrix
from causal_discovery.graph_discovery import GraphDiscovery
from causal_graphs.graph_generation import generate_categorical_graph, generate_random_graph, get_graph_func
from causal_graphs.graph_visualization import visualize_graph
from experiments.utils import set_cluster, test_graph


def create_graph(num_vars, num_categs, edge_prob, use_nn, graph_type, seed):
    graph = generate_categorical_graph(num_vars=num_vars,
                                       min_categs=num_categs,
                                       max_categs=num_categs,
                                       edge_prob=edge_prob,
                                       connected=True,
                                       use_nn=use_nn,
                                       graph_func=get_graph_func(graph_type),
                                       seed=seed)
    return graph



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_vars', type=int, default=8)
    parser.add_argument('--num_categs', type=int, default=10)
    parser.add_argument('--num_graphs', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--edge_prob', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--model_iters', type=int, default=1000)
    parser.add_argument('--lambda_sparse', type=float, default=0.01)
    parser.add_argument('--theta_regularizer', action='store_true')
    parser.add_argument('--lr_model', type=float, default=2e-2)
    parser.add_argument('--lr_permut', type=float, default=5e-3)
    parser.add_argument('--lr_gamma', type=float, default=5e-3)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--order_grads_opt', type=str, default='weight_expect_no_denom_weight_intervention')
    parser.add_argument('--edge_grads_opt', type=str, default='weight_expect')
    parser.add_argument('--beta2_theta', type=float, default=0.999)
    parser.add_argument('--N_s', type=int, default=1)
    parser.add_argument('--C_s', type=int, default=200)
    parser.add_argument('--max_graph_stacking', type=int, default=200)
    parser.add_argument('--theta_pretraining', type=int, default=0)
    parser.add_argument('--N_s_same_graph', action='store_true')
    parser.add_argument('--theta_alternate', action='store_true')
    parser.add_argument('--pretrain_N_s', type=int, default=-1)
    parser.add_argument('--pretrain_C_s', type=int, default=-1)
    parser.add_argument('--pretrain_iters', type=int, default=-1)
    parser.add_argument('--use_nn', action="store_true")
    parser.add_argument('--actfn', type=str, default='leakyrelu')
    parser.add_argument('--graph_type', type=str, default='random')
    parser.add_argument('--share_embeds', action="store_true")
    parser.add_argument('--sparse_embeds', action="store_true")
    parser.add_argument('--data_parallel', action="store_true")
    parser.add_argument('--data_parallel_fitting', action="store_true")
    parser.add_argument('--use_adam_theta', action="store_true")
    parser.add_argument('--logging', action="store_true")

    args = parser.parse_args()

    current_date = datetime.now()
    if args.checkpoint_dir is None or len(args.checkpoint_dir) == 0:
        checkpoint_dir = "checkpoints/%02d_%02d_%02d__%02d_%02d_%02d/" % (current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute, current_date.second)
    else:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    torch.manual_seed(args.seed)
    set_cluster(args.cluster)

    for gindex in range(args.num_graphs):
        graph = create_graph(num_vars=args.num_vars, 
                             num_categs=args.num_categs, 
                             edge_prob=args.edge_prob, 
                             use_nn=args.use_nn, 
                             graph_type=args.graph_type,
                             seed=args.seed+gindex)
        if graph.num_vars < 40:
            figsize = max(3, graph.num_vars/1.5)
        else:
            figsize = graph.num_vars**0.7
        if graph.num_vars <= 100:
            visualize_graph(graph, 
                            filename=os.path.join(checkpoint_dir, "graph_%i.pdf" % (gindex+1)),
                            figsize=(figsize, figsize), 
                            layout="circular" if graph.num_vars < 40 else "graphviz")
        graph.save_to_file(os.path.join(checkpoint_dir, "graph_%i.pt" % (gindex+1)))

        for DiscClass in [GraphDiscoveryMatrix]: # , GraphDiscoveryPairwise
            file_id = "%i_%s" % (gindex+1, DiscClass.__name__)
            test_graph(DiscClass, graph, args, checkpoint_dir, file_id)


