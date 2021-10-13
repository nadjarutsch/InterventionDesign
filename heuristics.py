import torch
import numpy as np
import argparse
from collections import defaultdict
import torch.nn.functional as F

from enco_model import AdjacencyMatrix
from datasets import GraphData
from causal_graphs.variable_distributions import _random_categ

@torch.no_grad()
def choose_intervention(args: argparse.Namespace, 
                        epoch: int, 
                        adj_matrix: AdjacencyMatrix,
                        true_adj: torch.Tensor) -> int:
    """Chooses a node for intervention.
    
    Args:
        args: Object from the argument parser that include the heuristic and
            temperature value.
        epoch: Current epoch (used for sequence heuristic).
        gamma: Tensor of shape (n_nodes, n_nodes) of gamma values (determining 
            edge probabilities).
        theta: Tensor of shape (n_nodes, n_nodes) of theta values (determining 
            edge directions).

    Returns:
        Index of the node to be intervened on.
    """    
    if args.log_heuristic == 'uniform':
        return uniform(adj_matrix.num_variables)
    
    elif args.log_heuristic == 'uncertain-outgoing':
        return uncertain_out(args, adj_matrix) 
    
    elif args.log_heuristic == 'sequence':
        return sequence(epoch, adj_matrix.num_variables) 
    
    elif args.log_heuristic == 'uncertain-children':
        return uncertain_children(args, adj_matrix) 
    
    elif args.log_heuristic == 'uncertain-neighbours':
        return uncertain_neighbours(args, adj_matrix) 
    
    elif args.log_heuristic == 'true-distance':
        return true_distance(args.temperature, true_adj, adj_matrix)
    
    elif args.log_heuristic == 'vary-uncertain':
        return vary_uncertain(args, adj_matrix, epoch)
    
    elif args.log_heuristic == 'num-children':
        return num_children(args, adj_matrix)
    
    elif args.log_heuristic == 'influence':
        return influence(args, adj_matrix)

@torch.no_grad()
def uniform(num_variables: int) -> int:
    """Samples an intervention node uniformly from all nodes (baseline)."""
    
    return np.random.randint(num_variables)

@torch.no_grad()
def uncertain_out(args: argparse.Namespace,
                  adj_matrix: AdjacencyMatrix) -> int:
    """More likely to intervene on nodes with highly uncertain outgoing edge."""
    
    uncertainty = adj_matrix.uncertainty(temperature=args.temperature).cpu() 
    int_idx = torch.multinomial(uncertainty.flatten(), num_samples=1)
    
    # outgoing edge
    return np.unravel_index(int_idx, (adj_matrix.num_variables, adj_matrix.num_variables))[0][0]


def sequence(epoch: int,
             num_variables: int) -> int:
    """Intervene sequentially on all nodes (baseline)."""
    
    return epoch % num_variables

@torch.no_grad()
def uncertain_children(args: argparse.Namespace,
                       adj_matrix: AdjacencyMatrix) -> int:
    """More likely to intervene on nodes where the sum of the uncertainty of 
    outgoing edges is high."""
    
    uncertainty = adj_matrix.uncertainty(temperature=args.temperature)    
    int_idx = torch.multinomial(torch.sum(uncertainty, 1), num_samples=1)
    
    return int_idx.item()

@torch.no_grad()
def num_children(args: argparse.Namespace,
                       adj_matrix: AdjacencyMatrix) -> int:
    
    num_children = torch.sum(adj_matrix.edge_probs(), 1).to(dtype=float)  
    int_idx = torch.multinomial(num_children, num_samples=1)
    
    return int_idx.item()
    
@torch.no_grad()
def uncertain_neighbours(args: argparse.Namespace,
                         adj_matrix: AdjacencyMatrix) -> int:
    """More likely to intervene on nodes where the sum of the uncertainty of 
    incoming and outgoing edges is high."""
    
    uncertainty = adj_matrix.uncertainty(temperature=args.temperature)
    
    incoming = torch.sum(uncertainty,0)
    outgoing = torch.sum(uncertainty,1)

    int_idx = torch.multinomial(incoming + outgoing, num_samples=1)
    
    return int_idx.item()


@torch.no_grad()
def influence(args: argparse.Namespace,
              adj_matrix: AdjacencyMatrix) -> int:
    """More likely to intervene on nodes with the most descendants."""
    
    edge_probs = adj_matrix.edge_probs()
    num_children = torch.sum(edge_probs, 1).to(dtype=float) 
    descendants = num_children
    for i in range(args.num_variables - 2):
        descendants += torch.einsum('ab,b->ab', edge_probs, num_children)
        edge_probs = torch.einsum('ab,ba->ab', edge_probs, edge_probs)
    
    print(descendants)
    int_idx = torch.multinomial(descendants, num_samples=1)
    
    return int_idx.item()


@torch.no_grad()
def vary_uncertain(args: argparse.Namespace,
                   adj_matrix: AdjacencyMatrix,
                   epoch: int) -> int:
    
    temperature = 2 * epoch / args.int_epochs - 1
    uncertainty = adj_matrix.uncertainty(temperature=temperature)
    
    incoming = torch.sum(uncertainty,0)
    outgoing = torch.sum(uncertainty,1)

    int_idx = torch.multinomial(incoming + outgoing, num_samples=1)
    
    return int_idx.item()

@torch.no_grad()
def true_distance(temperature: int,
                  true_adj: torch.Tensor, 
                  adj_matrix: AdjacencyMatrix) -> int:
    """Intervene on parent node of the edge with the highest distance to the
    ground truth (possible optimal intervention strategy)."""
    
    dist = F.softmax(temperature * (true_adj - adj_matrix.edge_probs()).abs())
    int_idx = torch.multinomial(dist.max(dim=1).values, num_samples=1)
    
    return int_idx.item()


def choose_distribution(args: argparse.Namespace,
                        obs_data: GraphData) -> np.array:
    
    if args.log_int_dist == 'uniform':
        return unif_dist(obs_data)
    
    elif args.log_int_dist == 'inverse-softmax':
        return inverse_softmax(args.log_temp_int, obs_data)
    

def unif_dist(obs_data: GraphData) -> list:
    int_dists = [_random_categ(size=n, scale=0.0, axis=-1) for n in obs_data.num_categs]
    return int_dists

def inverse_softmax(temperature: int, 
                    obs_data: GraphData) -> defaultdict:
    int_dists = defaultdict(list)
    for i, num_categs in enumerate(obs_data.num_categs):
        for n in range(num_categs):
            int_dists[i].append(1 / ((obs_data.data[:,i] == n).sum(dim=0).item() + 1)) # smoothing
        int_dists[i] = F.softmax(temperature * torch.Tensor(int_dists[i]))
    return int_dists
