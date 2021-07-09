import torch
import numpy as np
import argparse

from enco_model import AdjacencyMatrix


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
    if args.heuristic == 'uniform':
        return uniform(adj_matrix.num_variables)
    
    elif args.heuristic == 'uncertain-outgoing':
        return uncertain_out(args, adj_matrix) 
    
    elif args.heuristic == 'sequence':
        return sequence(epoch, adj_matrix.num_variables) 
    
    elif args.heuristic == 'uncertain-children':
        return uncertain_children(args, adj_matrix) 
    
    elif args.heuristic == 'uncertain-neighbours':
        return uncertain_neighbours(args, adj_matrix) 
    
    elif args.heuristic == 'true-distance':
        return true_distance(true_adj, adj_matrix)


def uniform(num_variables: int) -> int:
    """Samples an intervention node uniformly from all nodes (baseline)."""
    
    return np.random.randint(num_variables)


def uncertain_out(args: argparse.Namespace,
                  adj_matrix: AdjacencyMatrix) -> int:
    """More likely to intervene on nodes with highly uncertain outgoing edge."""
    
    uncertainty = adj_matrix.uncertainty(temperature=args.temperature)
    
    # don't sample variables based on self-cycle edges
    uncertainty.fill_diagonal_(0)
    
    int_idx = torch.multinomial(uncertainty.flatten(), num_samples=1)
    
    # outgoing edge
    return np.unravel_index(int_idx, adj_matrix.num_variables)[0][0]


def sequence(epoch: int,
             num_variables: int) -> int:
    """Intervene sequentially on all nodes (baseline)."""
    
    return epoch % num_variables


def uncertain_children(args: argparse.Namespace,
                       adj_matrix: AdjacencyMatrix) -> int:
    """More likely to intervene on nodes where the sum of the uncertainty of 
    outgoing edges is high."""
    
    uncertainty = adj_matrix.uncertainty(temperature=args.temperature)
    
    # don't sample variables based on self-cycle edges
    uncertainty.fill_diagonal_(0)
    
    int_idx = torch.multinomial(torch.sum(uncertainty, 1), num_samples=1)
    
    return int_idx
    

def uncertain_neighbours(args: argparse.Namespace,
                         adj_matrix: AdjacencyMatrix) -> int:
    """More likely to intervene on nodes where the sum of the uncertainty of 
    incoming and outgoing edges is high."""
    
    uncertainty = adj_matrix.uncertainty(temperature=args.temperature)
    
    # don't sample variables based on self-cycle edges
    uncertainty.fill_diagonal_(0)
    
    incoming = torch.sum(uncertainty,0)
    outgoing = torch.sum(uncertainty,1)

    int_idx = torch.multinomial(incoming + outgoing, num_samples=1)
    
    return int_idx


def true_distance(true_adj: torch.Tensor, 
                  adj_matrix: AdjacencyMatrix) -> int:
    """Intervene on parent node of the edge with the highest distance to the
    ground truth (possible optimal intervention strategy)."""
    
    dist = (true_adj - adj_matrix.edge_probs()).abs()
    int_idx = torch.argmax(dist.max(dim=1).values, dim=0).item()
    
    return int_idx
