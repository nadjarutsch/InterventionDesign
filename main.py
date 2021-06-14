from env import CausalEnv
from heuristics import choose_intervention
from graph_update import GraphUpdate # TODO

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from causal_discovery.multivariable_mlp import create_model, MultivarMLP
from experiments.utils import track
from causal_discovery.graph_fitting import GraphFitting
from DAG_matrix.adam_theta import AdamTheta


    
def main(args: argparse.Namespace):
    """Executes a causal discovery algorithm on synthetic data from a sampled
    DAG, using a specified heuristic for choosing intervention variables.
    
    Args:
        args: Object from the argument parser that defines various settings of
            the causal structure and discovery process.
    """
    # initialize model of the causal structure
    model, gamma, theta = init_model(args)
    
    # initialize optimizers
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model, betas=args.betas_model)
    
    if args.betas_gamma[1] > 0:
        gamma_optimizer = torch.optim.Adam([gamma], lr=args.lr_gamma, betas=args.betas_gamma)
    else:
        gamma_optimizer = torch.optim.SGD([gamma], lr=args.lr_gamma, momentum=args.betas_gamma[0])

    theta_optimizer = AdamTheta(theta, lr=args.lr_theta, beta1=args.betas_theta[0], beta2=args.betas_theta[1])
    
    # initialize the environment: create a graph and generate observational 
    # samples from the joint distribution of the graph
    env = CausalEnv(num_vars=args.num_variables, 
                    min_categs=args.min_categories,
                    max_categs=args.max_categories,
                    graph_structure=args.graph_structure)
    obs_data = env.reset(n_samples=args.n_obs_samples)
    obs_dataloader = DataLoader(obs_data, batch_size=args.obs_batch_size, shuffle=True, drop_last=True)
    
    # initialize fitting module
    distributionFitting = GraphFitting(model, 
                                 model_optimizer, 
                                 obs_dataloader)
    
    updateModule = GraphUpdate(gamma,
                               theta,
                               gamma_optimizer,
                               theta_optimizer)
    
    shd = eval_model() # TODO: structural hamming distance
    
    # causal discovery training loop
    for epoch in track(range(args.max_interventions), leave=False, desc="Epoch loop"):
        # fit model to observational data (distribution fitting)
        distributionFitting, loss = obs_step(args, distributionFitting, updateModule.gamma, updateModule.theta, obs_dataloader)
        
        # perform intervention and update parameters based on interventional data
        int_idx = choose_intervention(heuristic=args.heuristic, gamma=gamma.detach(), theta=theta.detach())
        int_data, reward, info = env.step(int_idx, args.n_int_samples) 
        int_dataloader = DataLoader(int_data, batch_size=args.int_batch_size, shuffle=True, drop_last=True)
        updateModule.dataloader = int_dataloader
       
        # graph fitting
        updateModule.epoch = epoch
        updateModule = int_step(args, updateModule, distributionFitting.model, int_idx)
        
        # TODO
        shd = eval_model()        
 #       metric_diff = metric_before - metric_after
  #      metric_rel = metric_after / metric_before 



   
def init_model(args: argparse.Namespace) -> Tuple[MultivarMLP, nn.Parameter, nn.Parameter]:
    """Initializes a complete model of the causal structure, consisting of a 
    multivariable MLP which models the conditional distributions of the causal 
    variables, and gamma and theta values which define the adjacency matrix 
    of the causal graph.
    
    Args:
        args: Object from the argument parser that defines various settings of
            the causal structure and discovery process.
    
    Returns:
        model: Multivariable MLP (modelling the conditional distributions).
        gamma: Matrix of gamma values (determining edge probabilities).
        theta: Matrix of theta values (determining edge directions).
    """
    model = create_model(num_vars=args.num_variables, 
                         num_categs=args.max_categories, 
                         hidden_dims=[64], 
                         share_embeds=False,
                         actfn='leakyrelu',
                         sparse_embeds=False)
    if args.data_parallel:
            print("Data parallel activated. Using %i GPUs" % torch.cuda.device_count())
            model = nn.DataParallel(model)
    
    gamma = nn.Parameter(torch.zeros(args.num_variables, args.num_variables)) 
    gamma.data[torch.arange(args.num_variables), torch.arange(args.num_variables)] = -9e15
    theta = nn.Parameter(torch.zeros(args.num_variables,args.num_variables))
    
    return model, gamma, theta

    
def obs_step(args: argparse.Namespace,
             distributionFitting: GraphFitting,
             gamma: nn.Parameter,
             theta: nn.Parameter,
             dataloader: DataLoader) -> Tuple[GraphFitting, float]:
    """Fit the multivariable MLP to observational data, given the predicted
    adjacency matrix.
    
    Args:
        args: Object from the argument parser that defines various settings of
            the causal structure and discovery process.
        distributionFitting: Module used for fitting the MLP to observational data 
            and the predicted adjacency matrix.
        gamma: Matrix of gamma values (determining edge probabilities).
        theta: Matrix of theta values (determining edge directions).
        dataloader: Dataloader with observational data from the joint distri-
            bution.
    
    Returns:
        distributionFitting: Fitting module with updated MLP.
        loss: Average loss in one fitting epoch.
    """
    sample_matrix = torch.sigmoid(gamma.detach())
    sfunc = lambda batch_size: sample_func(sample_matrix=sample_matrix, 
                                           theta=theta,
                                           batch_size=args.obs_batch_size) 
        
    avg_loss = 0.0
    t = track(range(args.fitting_epochs), leave=False, desc="Model update loop")
    for _ in t:
        loss = distributionFitting.fit_step(sample_func=sfunc)
        avg_loss += loss
        if hasattr(t, "set_description"):
            t.set_description("Model update loop, loss: %4.2f" % loss)

    avg_loss /= args.fitting_epochs 
    
    return distributionFitting, avg_loss


def int_step(args: argparse.Namespace,
             updateModule: GraphUpdate,
             model: MultivarMLP,
             int_idx: int) -> GraphUpdate:

    for _ in track(range(args.int_epochs), leave=False, desc="Gamma and theta update loop"):
        updateModule.update_step(model, int_idx) 

    return updateModule

    
def sample_func(sample_matrix, theta, batch_size):
        A = sample_matrix[None].expand(batch_size, -1, -1)
        A = torch.bernoulli(A)
        order_p = torch.sigmoid(theta) * (1 - torch.eye(theta.shape[0], device=theta.device))
        order_p = order_p[None].expand(batch_size, -1, -1)
        order_mask = torch.bernoulli(order_p)

        A = A * order_mask

        return A 
    
# TODO
def eval_model():
    return 1


    
 
if __name__ == '__main__':
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('--data_parallel', default=False, type=bool, help='Use parallelization for efficiency')
    parser.add_argument('--num_variables', default=4, type=int, help='Number of causal variables')
    parser.add_argument('--min_categories', default=2, type=int, help='Minimum number of categories of a causal variable')
    parser.add_argument('--max_categories', default=10, type=int, help='Maximum number of categories of a causal variable')
    parser.add_argument('--n_obs_samples', default=10000, type=int, help='Number of observational samples from the joint distribution of a synthetic graph')
    parser.add_argument('--n_int_samples', default=1000, type=int, help='Number of samples from one intervention')
    parser.add_argument('--max_interventions', default=1, type=int, help='Maximum number of interventions')
    parser.add_argument('--graph_structure', choices=['random', 'jungle', 'chain'], default='random', help='Structure of the true causal graph')
    parser.add_argument('--heuristic', choices=['uniform', 'uncertain'], default='uncertain', help='Heuristic used for choosing intervention nodes')

    # Graph fitting (observational data)
    parser.add_argument('--obs_batch_size', default=128, type=int, help='Batch size used for fitting the graph to observational data')
    parser.add_argument('--fitting_epochs', default=10, type=int, help='Number of epochs for fitting the causal structure to observational data')
    
    # Optimizers
    parser.add_argument('--lr_model', default=2e-2, type=float, help='Learning rate for fitting the model to observational data')
    parser.add_argument('--betas_model', default=(0.9,0.999), type=tuple, help='Betas used for Adam optimizer (model fitting)')
    parser.add_argument('--lr_gamma', default=5e-3, type=float, help='Learning rate for updating gamma parameters')
    parser.add_argument('--betas_gamma', default=(0.1,0.1), type=tuple, help='Betas used for Adam optimizer OR momentum used for SGD (gamma update)')
    parser.add_argument('--lr_theta', default=5e-3, type=float, help='Learning rate for updating theta parameters')
    parser.add_argument('--betas_theta', default=(0.9,0.999), type=tuple, help='Betas used for Adam Theta optimizer (theta update)')
    
    # Graph update (interventional data)
    parser.add_argument('--int_batch_size', default=64, type=int, help='Batch size used for scoring based on interventional data')
    parser.add_argument('--int_epochs', default=10, type=int, help='Number of epochs for updating the graph gamma and theta parameters of the graph')
    

    args: argparse.Namespace = parser.parse_args()

    # run main loop
    main(args)