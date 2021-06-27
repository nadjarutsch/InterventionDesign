import torch
import numpy as np

HEURISTICS = ['uniform', 'uncertain']


def choose_intervention(heuristic: str, gamma: torch.Tensor, theta: torch.Tensor) -> int:
    """Chooses a node for intervention.
    
    Args:
        gamma_matrix: Tensor of shape (n_nodes, n_nodes) with gamma-parameters,
            representing the edge probabilities.

    Returns:
        Index of the node to be intervened on.
        
    Raises:
        Exception: If heuristic is not available.
    """    
    if heuristic == HEURISTICS[0]:
        return uniform(gamma)
    
    elif heuristic == HEURISTICS[1]:
        return uncertain(gamma, theta)  
    
    else:
        raise Exception('Heuristic is not available. \n Chosen heuristic: {} \n Available heuristics: {}'.format(heuristic, HEURISTICS))


def uniform(gamma: torch.Tensor) -> int:
    """Samples an intervention node uniformly from all nodes."""
    
    return np.random.randint(gamma.shape[-1])


def uncertain(gamma: torch.Tensor, theta: torch.Tensor) -> int:
    """Chooses the intervention node with the most uncertain outgoing edge."""
    
    # TODO: check if this works as intended, maybe try abs(gamma) + abs(theta)?
    certainty = gamma * theta
    print('CERTAINTY :', np.abs(certainty.numpy()), np.argmin(np.abs(certainty.numpy())))
    
    # TODO: check if [0] really returns the node with the most uncertain 
    # outgoing edge
    return np.unravel_index(np.argmin(np.abs(certainty.numpy())), gamma.shape)[1]
