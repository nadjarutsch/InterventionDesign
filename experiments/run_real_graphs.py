import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from datetime import datetime
import json
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from glob import glob
import os
import sys
sys.path.append("../")

from DAG_generation.graph_discovery import GraphDiscoveryDAG
from DAG_pairwise.graph_discovery import GraphDiscoveryPairwise
from DAG_matrix.graph_discovery import GraphDiscoveryMatrix
from causal_discovery.graph_discovery import GraphDiscovery
from causal_graphs.graph_real_world import load_graph_file
from experiments.utils import set_cluster, test_graph


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lambda_sparse', type=float, default=0.001)
    parser.add_argument('--lr_permut', type=float, default=5e-3)
    parser.add_argument('--lr_gamma', type=float, default=5e-3)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--order_grads_opt', type=str, default='weight_expect_no_denom_weight_intervention')
    parser.add_argument('--edge_grads_opt', type=str, default='weight_expect')
    parser.add_argument('--beta2_theta', type=float, default=0.999)
    parser.add_argument('--N_s', type=int, default=1)
    parser.add_argument('--C_s', type=int, default=200)
    parser.add_argument('--max_graph_stacking', type=int, default=200)
    parser.add_argument('--use_only', type=str, default='')
    parser.add_argument('--exclude', type=str, default='', nargs='+')

    args = parser.parse_args()

    current_date = datetime.now()
    if args.checkpoint_dir is None or len(args.checkpoint_dir) == 0:
        checkpoint_dir = "checkpoints/real_%02d_%02d_%02d__%02d_%02d_%02d/" % (current_date.year, current_date.month, current_date.day, current_date.hour, current_date.minute, current_date.second)
    else:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    set_cluster(args.cluster)

    if len(args.use_only) == 0:
        real_world_files = sorted(glob("../causal_graphs/real_data/*.bif"))
    else:
        real_world_files = ["../causal_graphs/real_data/%s.bif" % args.use_only]

    for gindex, filename in enumerate(real_world_files):
        graph = load_graph_file(filename)
        dataset_name = filename.split("/")[-1].split(".bif")[0]
        if dataset_name in args.exclude or (len(args.use_only) == 0 and dataset_name.startswith("test")):
            continue
        print("="*50 + "\nTesting dataset \"%s\"\n" % dataset_name + "="*50)

        for DiscClass in [GraphDiscoveryMatrix]:
            for seed_idx in range(args.num_seeds):
                pl.seed_everything(args.base_seed + seed_idx)
                file_id = "%s_%s_seed%i" % (dataset_name, DiscClass.__name__, args.base_seed + seed_idx)
                test_graph(DiscClass, graph, args, checkpoint_dir, file_id)


