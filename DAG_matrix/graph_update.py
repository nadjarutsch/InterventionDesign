import torch
import torch.nn.functional as F
import numpy as np 
import time
import sys
sys.path.append("../")


class GraphUpdateMatrix(object):


    def __init__(self, lambda_sparse, theta_regularizer=False, **kwargs):
        self.lambda_sparse = lambda_sparse
        self.theta_regularizer = theta_regularizer


    def update(self, adj_matrix, logregret, gamma, theta_matrix, var_idx):
        ## GAMMA UPDATE
        if gamma.grad is not None:
            gamma.grad.fill_(0)
        gamma_grads, theta_grads, debug = self.gradient_estimator(adj_matrix, logregret, theta_matrix, gamma, var_idx)
        gamma.grad = gamma_grads
        theta_matrix.grad = theta_grads

        return debug["theta_mask"]
    

    @torch.no_grad()
    def gradient_estimator(self, adj_matrix, logregret, theta_matrix, gamma, var_idx):
        batch_size = adj_matrix.shape[0]
        logregret = logregret.unsqueeze(dim=1)

        comp_probs = torch.sigmoid(theta_matrix)
        edge_probs = torch.sigmoid(gamma)
        
        num_pos = adj_matrix.sum(dim=0)
        num_neg = batch_size - num_pos
        mask = ((num_pos > 0) * (num_neg > 0)).float()
        pos_grads = (logregret * adj_matrix).sum(dim=0) / num_pos.clamp_(min=1e-5)
        neg_grads = (logregret * (1 - adj_matrix)).sum(dim=0) / num_neg.clamp_(min=1e-5)
        gamma_grads = mask * edge_probs * (1 - edge_probs) * comp_probs * (pos_grads - neg_grads + self.lambda_sparse)

        theta_inner_grads = edge_probs * (pos_grads - neg_grads)
        if self.theta_regularizer:
            theta_inner_grads += 0.5 * self.lambda_sparse * (edge_probs - edge_probs.T)
        theta_grads = mask * comp_probs * (1 - comp_probs) * theta_inner_grads
        theta_grads[:var_idx] = 0
        theta_grads[var_idx+1:] = 0
        theta_grads -= theta_grads.transpose(0, 1)

        # Creating a mask which theta's are actually updated
        theta_mask = torch.zeros_like(theta_grads)
        theta_mask[var_idx] = 1.
        theta_mask[:,var_idx] = 1.
        theta_mask[var_idx,var_idx] = 0

        debug = {
            "theta_grads": theta_grads,
            "comp_probs": comp_probs,
            "logregret": logregret,
            "theta_mask": theta_mask
        }
        
        return gamma_grads, theta_grads, debug