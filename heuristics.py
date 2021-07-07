import torch
import numpy as np
import argparse


def choose_intervention(args: argparse.Namespace, 
                        epoch: int, 
                        gamma: torch.Tensor, 
                        theta: torch.Tensor) -> int:
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
        return uniform(gamma)
    
    elif args.heuristic == 'uncertain-incoming':
        return uncertain_in(args, gamma, theta) 
    
    elif args.heuristic == 'uncertain-outgoing':
        return uncertain_out(args, gamma, theta) 
    
    elif args.heuristic == 'sequence':
        return sequence(args, epoch, gamma, theta) 
    
    elif args.heuristic == 'uncertain-children':
        return uncertain_children(args, gamma, theta) 
    
    elif args.heuristic == 'uncertain-parents':
        return uncertain_parents(args, gamma, theta) 
    
    elif args.heuristic == 'uncertain-neighbours':
        return uncertain_neighbours(args, gamma, theta) 


def uniform(gamma: torch.Tensor) -> int:
    """Samples an intervention node uniformly from all nodes (baseline)."""
    
    return np.random.randint(gamma.shape[-1])


def uncertain_out(args: argparse.Namespace,
              gamma: torch.Tensor, 
              theta: torch.Tensor) -> int:
    """More likely to intervene on nodes with highly uncertain outgoing edge."""
    
    probs = torch.sigmoid(args.temperature * gamma * theta) 
    certainty = probs * (1-probs)
    
    # don't sample variables based on self-cycle edges
    certainty.fill_diagonal_(0)
    
    int_idx = torch.multinomial(certainty.flatten(), num_samples=1)
    
    # outgoing edge
    return np.unravel_index(int_idx, gamma.shape)[0][0]


def uncertain_in(args: argparse.Namespace,
              gamma: torch.Tensor, 
              theta: torch.Tensor) -> int:
    """More likely to intervene on nodes with highly uncertain incoming edge."""
    
    probs = torch.sigmoid(args.temperature * gamma * theta) 
    certainty = probs * (1-probs)
    
    # don't sample variables based on self-cycle edges
    certainty.fill_diagonal_(0)
    
    int_idx = torch.multinomial(certainty.flatten(), num_samples=1)    

    return np.unravel_index(int_idx, gamma.shape)[1][0]


def sequence(args: argparse.Namespace,
             epoch: int,
             gamma: torch.Tensor, 
             theta: torch.Tensor) -> int:
    """Intervene sequentially on all nodes (baseline)."""
    
    return epoch % gamma.shape[0]


def uncertain_children(args: argparse.Namespace,
             gamma: torch.Tensor,
             theta: torch.Tensor) -> int:
    """More likely to intervene on nodes where the sum of the uncertainty of 
    outgoing edges is high."""
    
    probs = torch.sigmoid(args.temperature * gamma * theta) 
    certainty = probs * (1-probs)
    
    # don't sample variables based on self-cycle edges
    certainty.fill_diagonal_(0)
    
    int_idx = torch.multinomial(torch.sum(certainty, 1), num_samples=1)
    
    return int_idx
    
  
def uncertain_parents(args: argparse.Namespace,
            gamma: torch.Tensor,
            theta: torch.Tensor) -> int:
    """More likely to intervene on nodes where the sum of the uncertainty of 
    incoming edges is high."""
    
    probs = torch.sigmoid(args.temperature * gamma * theta) 
    certainty = probs * (1-probs)
    
    # don't sample variables based on self-cycle edges
    certainty.fill_diagonal_(0)

    int_idx = torch.multinomial(torch.sum(certainty, 0), num_samples=1)
    
    return int_idx


def uncertain_neighbours(args: argparse.Namespace,
            gamma: torch.Tensor,
            theta: torch.Tensor) -> int:
    """More likely to intervene on nodes where the sum of the uncertainty of 
    incoming and outgoing edges is high."""
    
    probs = torch.sigmoid(args.temperature * gamma * theta) 
    
    certainty = probs * (1-probs)
    
    # don't sample variables based on self-cycle edges
    certainty.fill_diagonal_(0)
    
    incoming = torch.sum(certainty,0)
    outgoing = torch.sum(certainty,1)

    int_idx = torch.multinomial(incoming + outgoing, num_samples=1)
    
    return int_idx
    
