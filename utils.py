from tqdm.auto import tqdm 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import time
from copy import deepcopy
import json
import os
import sys
sys.path.append("../")
from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.graph_utils import adj_matrix_to_edges

CLUSTER = False

def set_cluster(is_cluster):
    global CLUSTER 
    CLUSTER = is_cluster
    if CLUSTER:
        matplotlib.use('Agg')

def is_cluster():
    return CLUSTER

def track(range_iter, **kwargs):
    if not CLUSTER:
        return tqdm(range_iter, **kwargs)
    else:
        return range_iter

def test_graph(DiscClass, graph, args, checkpoint_dir, file_id):
    discovery_module = DiscClass(graph=graph, 
                                 model_iters=args.model_iters, 
                                 batch_size=args.batch_size,
                                 hidden_dims=[args.hidden_size],
                                 lambda_sparse=args.lambda_sparse,
                                 theta_regularizer=args.theta_regularizer,
                                 lr_model=args.lr_model,
                                 lr_permut=args.lr_permut,
                                 lr_gamma=args.lr_gamma,
                                 order_grads_opt=args.order_grads_opt,
                                 edge_grads_opt=args.edge_grads_opt,
                                 betas_theta=(0.9,args.beta2_theta),
                                 N_s=args.N_s,
                                 C_s=args.C_s,
                                 N_s_same_graph=args.N_s_same_graph,
                                 max_graph_stacking=args.max_graph_stacking,
                                 theta_pretraining=args.theta_pretraining,
                                 theta_alternate=args.theta_alternate,
                                 pretrain_N_s=args.pretrain_N_s,
                                 pretrain_C_s=args.pretrain_C_s,
                                 pretrain_iters=args.pretrain_iters,
                                 share_embeds=args.share_embeds,
                                 sparse_embeds=args.sparse_embeds,
                                 actfn=args.actfn,
                                 data_parallel=args.data_parallel,
                                 data_parallel_fitting=args.data_parallel_fitting,
                                 use_adam_theta=args.use_adam_theta)
    discovery_module.to("cuda:0")
    start_time = time.time()
    discovery_module.discover_graph(num_epochs=args.num_epochs)
    duration = int(time.time() - start_time)
    print("-> Finished training in %ih %imin %is" % (duration//3600, (duration//60)%60, duration%60))
    metrics = discovery_module.get_metrics()
    with open(os.path.join(checkpoint_dir, "metrics_%s.json" % file_id), "w") as f:
        json.dump(metrics, f, indent=4)

    if graph.num_vars < 40:
        pred_graph = deepcopy(graph)
        pred_graph.adj_matrix = discovery_module.get_binary_adjmatrix().detach().cpu().numpy()
        pred_graph.edges = adj_matrix_to_edges(pred_graph.adj_matrix)
        figsize = max(3, pred_graph.num_vars/1.5)
        visualize_graph(pred_graph, 
                        filename=os.path.join(checkpoint_dir, "graph_%s.pdf" % (file_id)), 
                        figsize=(figsize, figsize), 
                        layout="circular")

    if args.logging:
        if hasattr(discovery_module, "theta_log"):
            thetas = torch.stack(discovery_module.theta_log, dim=0).numpy()
            np.savez(os.path.join(checkpoint_dir, "thetas_%s.npz" % file_id), thetas)
            sns.set()
            for i in range(thetas.shape[1]):
                plt.plot(np.arange(thetas.shape[0]), thetas[:,i], label=r"$\theta_{%s}$" % (graph.variables[i].name))
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.xlabel("Training iterations")
            plt.ylabel("Logit value")
            plt.title("Training progress of order parameters")
            plt.savefig(os.path.join(checkpoint_dir, "logits_training_progress_%s.pdf" % file_id), bbox_inches='tight')
            plt.close()
        
        if hasattr(discovery_module, "theta_grad_log"):
            theta_grads = torch.stack(discovery_module.theta_grad_log, dim=0).numpy()
            np.savez(os.path.join(checkpoint_dir, "theta_grads_%s.npz" % file_id), theta_grads)
            sns.set()
            for i in range(thetas.shape[1]):
                plt.plot(np.arange(theta_grads.shape[0]), gaussian_filter1d(theta_grads[:,i], sigma=40), label=r"$\theta_{%s}$" % (graph.variables[i].name))
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.xlabel("Training iterations")
            plt.ylabel("Logit gradient")
            plt.title("Gradient values of order parameters")
            plt.savefig(os.path.join(checkpoint_dir, "logits_gradient_progress_%s.pdf" % file_id), bbox_inches='tight')
            plt.close()

        for attr_name in ["theta_matrix_log", "theta_matrix_grad_log", "gamma_log", "gamma_grad_log"]:
            if hasattr(discovery_module, attr_name):
                attr_val = torch.stack(getattr(discovery_module, attr_name), dim=0).numpy().astype(np.float16)
                np.savez(os.path.join(checkpoint_dir, "%s_%s.npz" % (attr_name, file_id)), attr_val)

    if hasattr(discovery_module, "metric_log"):
        with open(os.path.join(checkpoint_dir, "metrics_full_log_%s.json" % file_id), "w") as f:
            json.dump(discovery_module.metric_log, f, indent=4)

    if hasattr(discovery_module, "get_state_dict"):
        torch.save(discovery_module.get_state_dict(), 
                   os.path.join(checkpoint_dir, "state_dict_%s.tar" % file_id))