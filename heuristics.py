import torch
import numpy as np
import argparse

HEURISTICS = ['uniform', 'uncertain', 'sequence', 'children', 'parents']


def choose_intervention(args: argparse.Namespace, epoch: int, gamma: torch.Tensor, theta: torch.Tensor) -> int:
    """Chooses a node for intervention.
    
    Args:
        gamma_matrix: Tensor of shape (n_nodes, n_nodes) with gamma-parameters,
            representing the edge probabilities.

    Returns:
        Index of the node to be intervened on.
        
    Raises:
        Exception: If heuristic is not available.
    """    
    if args.heuristic == HEURISTICS[0]:
        return uniform(gamma)
    
    elif args.heuristic == HEURISTICS[1]:
        return uncertain(args, gamma, theta) 
    
    elif args.heuristic == HEURISTICS[2]:
        return sequence(args, epoch, gamma, theta) 
    
    elif args.heuristic == HEURISTICS[3]:
        return children(args, gamma, theta) 
    
    elif args.heuristic == HEURISTICS[4]:
        return parents(args, gamma, theta) 
    
    else:
        raise Exception('Heuristic is not available. \n Chosen heuristic: {} \n Available heuristics: {}'.format(args.heuristic, HEURISTICS))


def uniform(gamma: torch.Tensor) -> int:
    """Samples an intervention node uniformly from all nodes."""
    
    return np.random.randint(gamma.shape[-1])


def uncertain(args: argparse.Namespace,
              gamma: torch.Tensor, 
              theta: torch.Tensor) -> int:
    """Chooses the intervention node with the most uncertain outgoing edge."""
    
    probs = torch.sigmoid(args.temperature * gamma * theta) 
    certainty = probs * (1-probs)
    
    # don't sample variables based on self-cycle edges
    certainty.fill_diagonal_(0)
    
    int_idx = torch.multinomial(certainty.flatten(), num_samples=1)
    
    # TODO: check if [0] really returns the node with the most uncertain 
    # outgoing edge
    return np.unravel_index(int_idx, gamma.shape)[0][0]


def sequence(args: argparse.Namespace,
             epoch: int,
             gamma: torch.Tensor, 
             theta: torch.Tensor) -> int:
    """Intervene sequentially on all nodes."""
    
    return epoch % gamma.shape[0]


def children(args: argparse.Namespace,
             gamma: torch.Tensor,
             theta: torch.Tensor) -> int:
    "Intervene on node with the most uncertain children."
    
    probs = torch.sigmoid(args.temperature * gamma * theta) 
    certainty = probs * (1-probs)
    
    int_idx = torch.multinomial(torch.sum(certainty, 1), num_samples=1)
    
    return int_idx
    
  
def parents(args: argparse.Namespace,
            gamma: torch.Tensor,
            theta: torch.Tensor) -> int:
    "Intervene on node with the most uncertain parents."
    
    probs = torch.sigmoid(args.temperature * gamma * theta) 
    certainty = probs * (1-probs)
    
    int_idx = torch.multinomial(torch.sum(certainty, 2), num_samples=1)
    
    return int_idx
    
