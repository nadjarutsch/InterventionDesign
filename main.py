from env import CausalEnv
from metrics import *
from enco_model import *
from enco_training import *
from heuristics import *
from policy import MLP, GAT

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
from collections import defaultdict
import json
from datetime import datetime
from statistics import mean
import os

from utils import track
from causal_graphs.graph_generation import generate_categorical_graph, get_graph_func
from causal_graphs.graph_definition import CausalDAG






    
def main(args: argparse.Namespace, dag: CausalDAG=None, policy=None):
    """Executes a causal discovery algorithm on synthetic data from a sampled
    DAG, using a specified heuristic for choosing intervention variables.
    
    Args:
        args: Object from the argument parser that defines various settings of
            the causal structure and discovery process.
    """
    
    # initialize the environment: create a graph and generate observational 
    # samples from the joint distribution of the graph variables
    env = CausalEnv(num_vars=args.num_variables, 
                    min_categs=args.min_categories,
                    max_categs=args.max_categories,
                    graph_structure=args.graph_structure,
                    edge_prob=args.edge_prob,
                    dag=dag)
    
    obs_data = env.reset(n_samples=args.n_obs_samples)
    obs_dataloader = DataLoader(obs_data, batch_size=args.obs_batch_size, shuffle=True, drop_last=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'    
    
    # initialize policy learning
    if args.learn_policy:
        policy = MLP(args.num_variables, [512, 256, 128]).float()
        policy = policy.to(device)
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        rewards_lst = []
        
        for t in range(args.max_episodes):
            policy_optimizer.zero_grad()
            log_probs, reward = train(args, env, obs_dataloader, device, policy)
            
            reward += [0] * (args.epochs - len(reward))
            rewards_lst.append(reward)
            baseline = args.beta_baseline * torch.Tensor(reward) + (1 - args.beta_baseline) *  baseline if t != 0 else torch.Tensor(reward)

            policy_loss = -torch.sum((torch.Tensor(reward[:len(log_probs)]) - baseline[:len(log_probs)]) * torch.cumsum(torch.tensor(log_probs, requires_grad=True), dim=0))
            
            policy_loss.backward()
            policy_optimizer.step()
            
            print(torch.sum(torch.Tensor(reward)))
            print(torch.mean(torch.sum(torch.tensor(rewards_lst), dim=-1)))
            
            if torch.sum(torch.Tensor(reward)) >= max(torch.sum(torch.tensor(rewards_lst), dim=-1)):
                print('\nSaving policy...')
                torch.save(policy.state_dict(), 'policy_mlp.pth')
            
    else:
        train(args, env, obs_dataloader, device, policy)
            
            
def train(args, env, obs_dataloader, device, policy=None):
    # initialize model of the causal structure
    model, adj_matrix = init_model(args, device)
    
    # initialize optimizers
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_model, betas=args.betas_model)   
    gamma_optimizer = AdamGamma(adj_matrix.gamma, lr=args.lr_gamma, beta1=args.betas_gamma[0], beta2=args.betas_gamma[1])    
    theta_optimizer = AdamTheta(adj_matrix.theta, lr=args.lr_theta, beta1=args.betas_theta[0], beta2=args.betas_theta[1])
    
    obs_data = env.reset(n_samples=args.n_obs_samples)
    obs_dataloader = DataLoader(obs_data, batch_size=args.obs_batch_size, shuffle=True, drop_last=True)
    int_dists = choose_distribution(args, obs_data)
    
    # initialize CE loss module 
    loss_module = nn.CrossEntropyLoss()
    
    # initialize Logger
    logger = Logger(args)
    logger.before_training(adj_matrix, env.dag)
    
    log_probs_lst = []
    reward_lst = []
    
    distance = torch.sum(torch.abs(torch.from_numpy(env.dag.adj_matrix).float().to(device) - adj_matrix.edge_probs()))
    
    # causal discovery training loop
    for epoch in track(range(args.epochs), leave=False, desc="Epoch loop"):
        # fit model to observational data (distribution fitting)
        avg_loss = distribution_fitting(args, 
                                        model, 
                                        loss_module, 
                                        model_optimizer, 
                                        adj_matrix, 
                                        obs_dataloader)    
       
        # graph fitting
        log_probs, reward = graph_fitting(args, 
                                          adj_matrix, 
                                          gamma_optimizer, 
                                          theta_optimizer, 
                                          model, 
                                          env,
                                          epoch,
                                          logger,
                                          int_dists,
                                          policy)
        
        distance_new = torch.sum(torch.abs(torch.from_numpy(env.dag.adj_matrix).float().to(device) - adj_matrix.edge_probs()))
        
        reward, distance = reward / 10. + (distance - distance_new), distance_new
        
        log_probs_lst.append(log_probs)
        reward_lst.append(reward.detach().item())
        
        # logging
        stop = logger.on_epoch_end(adj_matrix, torch.from_numpy(env.dag.adj_matrix), epoch)
        
        # stop early if SHD is 0 for 3 epochs
        if stop:
            break
    
    return log_probs_lst, reward_lst

   
def init_model(args: argparse.Namespace, device) -> Tuple[MultivarMLP, AdjacencyMatrix]:
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
                         hidden_dims=args.hidden_dims, 
                         actfn='leakyrelu')
    
    
    if args.data_parallel:
        device = torch.device("cuda:0")
        print("Data parallel activated. Using %i GPUs" % torch.cuda.device_count())
        model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")
    
    adj_matrix = AdjacencyMatrix(args.num_variables, device)
    return model, adj_matrix
    
 
if __name__ == '__main__':
    
    # collect cmd line args
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('--data_parallel', default=True, type=bool, help='Use parallelization for efficiency')
    parser.add_argument('--num_variables', default=25, type=int, help='Number of causal variables')
    parser.add_argument('--min_categories', default=10, type=int, help='Minimum number of categories of a causal variable')
    parser.add_argument('--max_categories', default=10, type=int, help='Maximum number of categories of a causal variable')
    parser.add_argument('--n_obs_samples', default=100000, type=int, help='Number of observational samples from the joint distribution of a synthetic graph')
    parser.add_argument('--epochs', default=30, type=int, help='Maximum number of interventions')
    parser.add_argument('--graph_structure', type=str, nargs='+', default=['chain'], help='Structure of the true causal graph')
    parser.add_argument('--heuristic', type=str, nargs='+', default=['uniform'], help='Heuristic used for choosing intervention nodes')
    parser.add_argument('--temperature', default=10.0, type=float, help='Temperature used for sampling the intervention variable')
    parser.add_argument('--full_test', default=True, type=bool, help='Full test run for comparison of all heuristics (fixed graphs)')
    parser.add_argument('--edge_prob', default=0.4, help='Edge likelihood for generating a graph') # only used for "random" graph structure
    parser.add_argument('--num_graphs', default=1, type=int, help='Number of graphs per structure')
    parser.add_argument('--existing_dags', dest='existing_dags', action='store_true')
    parser.add_argument('--generate_dags', dest='existing_dags', action='store_false')
    parser.set_defaults(existing_dags=True)

    # Distribution fitting (observational data)
    parser.add_argument('--obs_batch_size', default=128, type=int, help='Batch size used for fitting the graph to observational data')
    parser.add_argument('--obs_epochs', default=1000, type=int, help='Number of epochs for fitting the causal structure to observational data')
    parser.add_argument('--hidden_dims', default=[64], type=list, nargs='+', help='Number of hidden units in each layer of the Multivariable MLP')
    
    # Optimizers
    parser.add_argument('--lr_model', default=5e-3, type=float, help='Learning rate for fitting the model to observational data')
    parser.add_argument('--betas_model', default=(0.9,0.999), type=tuple, help='Betas used for Adam optimizer (model fitting)')
    parser.add_argument('--lr_gamma', default=2e-2, type=float, help='Learning rate for updating gamma parameters')
    parser.add_argument('--betas_gamma', default=(0.9,0.9), type=tuple, help='Betas used for Adam optimizer OR momentum used for SGD (gamma update)')
    parser.add_argument('--lr_theta', default=1e-1, type=float, help='Learning rate for updating theta parameters')
    parser.add_argument('--betas_theta', default=(0.9,0.999), type=tuple, help='Betas used for Adam Theta optimizer (theta update)')
    
    # Graph fitting (interventional data)
    parser.add_argument('--int_batch_size', default=128, type=int, help='Number of samples per intervention')
    parser.add_argument('--int_epochs', default=100, type=int, help='Number of epochs for updating the graph gamma and theta parameters of the graph')
    parser.add_argument('--int_dist', type=str, nargs='+', default=['uniform'], help='Categorical distribution used for sampling intervention values')
    parser.add_argument('--lambda_sparse', default=0.004, type=float, help='Threshold for interpreting an edge as beneficial')
    parser.add_argument('--K', default=100, help='Number of graph samples for gradient estimation')
    parser.add_argument('--temp_int', default=[1], type=float, nargs='+', help='Temperature used for distribution of intervention values')
    
    # Reinforcement Learning
    parser.add_argument('--max_episodes', default=10000, type=int, help='Maximum number of episodes')
    parser.add_argument('--learn_policy', dest='learn_policy', action='store_true')
    parser.add_argument('--beta_baseline', default=0.5, type=float, help='Beta used for exponentialy weighted baseline average')
    parser.set_defaults(learn_policy=True)

    args: argparse.Namespace = parser.parse_args()

    # test runs to compare different heuristics on the same graphs
    if args.full_test:
        dags = defaultdict(list)
        argparse_dict = vars(args)
        with open(datetime.today().strftime('tb_logs/%Y-%m-%d-%H-%M-hparams.json'), 'w') as fp:
            json.dump(argparse_dict, fp)
        for structure in args.graph_structure:
            for i in range(args.num_graphs):
                dag = generate_categorical_graph(num_vars=args.num_variables,
                                                 min_categs=args.min_categories,
                                                 max_categs=args.max_categories,
                                                 connected=True,
                                                 graph_func=get_graph_func(structure),
                                                 edge_prob=args.edge_prob,
                                                 use_nn=True)
                
                if args.existing_dags:
                    for root, dirs, files in os.walk('dags'):
                        if structure not in root:
                            continue
                        if f'dag-{i}' not in root:
                            continue
                        for file in files:
                            if 'dag.pt' in file:
                                path = os.path.join(root, file)
                                break
                        else:
                            continue
                        break
                    
                    dag = dag.load_from_file(path)
                    
                dags[structure].append(dag)
            
       #     for heuristic in args.heuristic:
            for int_dist in args.int_dist:
                if int_dist == 'uniform':
                    temp_int = [1]
                else:
                    temp_int = args.temp_int
                for temperature in temp_int:                           
                    for i, dag in enumerate(dags[structure]):
                        args.log_graph_structure = structure + "-dag-" + str(i)  # for logging
                        args.log_heuristic = 'mlp-policy' # for logging 
                        args.log_temp_int = temperature
                        args.log_int_dist = int_dist
                        main(args, dag)