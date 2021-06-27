import torch
import numpy as np
import argparse

HEURISTICS = ['uniform', 'uncertain']


def choose_intervention(args: argparse.Namespace, gamma: torch.Tensor, theta: torch.Tensor) -> int:
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
