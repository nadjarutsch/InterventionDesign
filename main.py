from env import CausalEnv
from heuristics import choose_intervention

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

from causal_discovery.multivariable_mlp import create_model, MultivarMLP
from experiments.utils import track
from causal_discovery.graph_fitting import GraphFitting


    
def main(args: argparse.Namespace):
    """Executes a causal discovery algorithm on synthetic data from a sampled
    DAG, using a specified heuristic for choosing intervention variables.
    
    Args:
        args: Object from the argument parser that defines various settings of
            the causal structure and discovery process.
    """
    # initialize model of the causal structure
    model, gamma, theta  = init_model(args)
    
    # initialize optimizers
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model, betas=args.betas_model)
    
    # initialize the environment: create a graph and generate observational 
    # samples from the joint distribution of the graph
    env = CausalEnv(num_vars=args.num_variables, 
                    max_categs=args.num_categories,
                    graph_structure=args.graph_structure)
    obs_data = env.reset(n_samples=args.n_obs_samples)
    obs_dataloader = DataLoader(obs_data, batch_size=args.obs_batch_size, shuffle=True, drop_last=True)
    
    # initialize discovery modules
    fittingModule = GraphFitting(model, 
                                 model_optimizer, 
                                 obs_dataloader)
    
    # causal discovery training loop
    for epoch in track(range(args.max_interventions), leave=False, desc="Epoch loop"):
        # fit model to observational data
        fittingModule, loss = obs_step(args, fittingModule, gamma, theta, obs_dataloader)
        
        metric_before = eval_model() # TODO: needed?
        
        # TODO
        # perform intervention and update parameters based on interventional data
        int_idx = choose_intervention(heuristic=args.heuristic, gamma=gamma.detach())
        int_step()
        
        # TODO
        metric_after = eval_model()        
        metric_diff = metric_before - metric_after
        metric_rel = metric_after / metric_before 




        
def init_model(args: argparse.Namespace) -> Tuple[MultivarMLP, nn.Parameter, nn.Parameter]:
    """Initializes a complete model of the causal structure, consisting of a 
    multivariable MLP which models the conditional distributions of the causal 
    variables, and gamma and theta values which describe the adjacency matrix 
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
                         num_categs=args.num_categories, 
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
             fittingModule: GraphFitting,
             gamma: nn.Parameter,
             theta: nn.Parameter,
             dataloader: DataLoader) -> Tuple[GraphFitting, float]:
    """Fit the multivariable MLP to observational data, given the predicted
    adjacency matrix.
    
    Args:
        args: Object from the argument parser that defines various settings of
            the causal structure and discovery process.
        fittingModule: Module used for fitting the MLP to observational data 
            and the predicted adjacency matrix.
        gamma: Matrix of gamma values (determining edge probabilities).
        theta: Matrix of theta values (determining edge directions).
        dataloader: Dataloader with observational data from the joint distri-
            bution.
    
    Returns:
        fittingModule: Fitting module with updated MLP.
        loss: Average loss in one fitting epoch.
    """
    sample_matrix = torch.sigmoid(gamma.detach())
    sfunc = lambda batch_size: sample_func(sample_matrix=sample_matrix, 
                                           theta=theta,
                                           batch_size=args.obs_batch_size) 
        
    avg_loss = 0.0
    t = track(range(args.fitting_epochs), leave=False, desc="Model update loop")
    for _ in t:
        loss = fittingModule.fit_step(sample_func=sfunc)
        avg_loss += loss
        if hasattr(t, "set_description"):
            t.set_description("Model update loop, loss: %4.2f" % loss)

    avg_loss /= args.fitting_epochs # TODO: do I need this?
    
    return fittingModule, avg_loss

    
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


# TODO
def int_step():
    pass


    
 
if __name__ == '__main__':
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('--data_parallel', default=False, type=bool, help='Use parallelization for efficiency')
    parser.add_argument('--num_variables', default=4, type=int, help='Number of causal variables')
    parser.add_argument('--num_categories', default=10, type=int, help='Maximum number of categories of a causal variable')
    parser.add_argument('--n_obs_samples', default=10000, type=int, help='Number of observational samples from the joint distribution of a synthetic graph')
    parser.add_argument('--max_interventions', default=1, type=int, help='Maximum number of interventions')
    parser.add_argument('--graph_structure', choices=['random', 'jungle', 'chain'], default='random', help='Structure of the true causal graph')
    parser.add_argument('--heuristic', choices=['uniform', 'uncertain'], default='uncertain', help='Heuristic used for choosing intervention nodes')

    # Graph fitting
    parser.add_argument('--lr_model', default=2e-2, type=float, help='Learning rate for fitting the model to observational data')
    parser.add_argument('--betas_model', default=(0.9,0.999), type=tuple, help='Betas used for Adam optimizer')
    parser.add_argument('--obs_batch_size', default=128, type=int, help='Batch size used for fitting the graph to observational data')
    parser.add_argument('--fitting_epochs', default=10, type=int, help='Number of epochs for fitting the causal structure to observational data' )

    args: argparse.Namespace = parser.parse_args()

    # run main loop
    main(args)