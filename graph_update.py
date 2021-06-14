import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append("../")
import math

from experiments.utils import track


class GraphUpdate(object):
    def __init__(self, gamma, 
                 theta, 
                 gamma_optimizer, 
                 theta_optimizer, 
                 theta_pretraining=False,
                 theta_alternate=False,
                 gamma_iters=1,
                 max_graph_stacking=1000,
                 lambda_sparse=0.001,
                 theta_regularizer=False):
        super().__init__()
        self.gamma = gamma
        self.theta = theta
        self.gamma_optimizer = gamma_optimizer
        self.theta_optimizer = theta_optimizer
        self.epoch = 0
        self.theta_pretraining = theta_pretraining
        self.theta_alternate = theta_alternate
        self.gamma_iters = gamma_iters
        self.pretrain_iters = 10*self.gamma_iters
        self.max_graph_stacking = max_graph_stacking
        self.lambda_sparse=lambda_sparse
        self.theta_regularizer = theta_regularizer
        
        
        
    def update_step(self, model, int_idx):
        self.data_iter = iter(self.dataloader)
        only_theta = (self.epoch < self.theta_pretraining) or (self.theta_alternate and self.epoch % 2 == 0)
        iters = self.gamma_iters if not only_theta else self.pretrain_iters
        for _ in track(range(iters), leave=False, desc="Gamma and theta update loop"):
            self.gamma_optimizer.zero_grad()
            self.theta_optimizer.zero_grad()
            int_batch = self._get_next_batch()
            adj_matrices, logregret = self.score(int_batch, int_idx, model, only_theta)
            theta_mask = self.update(int_idx, adj_matrices, logregret) # TODO
            print('INT_IDX ', int_idx)
            if not only_theta:
                self.gamma_optimizer.step()
            self.theta_optimizer.step(theta_mask)
                
                
    def score(self, int_batch, 
              int_idx, 
              model, 
              only_theta, 
              mirror_graphs=False, 
              C_s=200):

        adj_matrices = []
        logregret = []

        C_s_list = [min(self.max_graph_stacking, C_s-i*self.max_graph_stacking) for i in range(math.ceil(C_s * 1.0 / self.max_graph_stacking))]
        C_s_list = [(C_s_list[i],sum(C_s_list[:i])) for i in range(len(C_s_list))]

        edge_prob = torch.sigmoid(self.gamma).detach()
        orientation_prob = torch.sigmoid(self.theta).detach()
        edge_prob_batch = edge_prob[None].expand(C_s,-1,-1)
        orientation_prob_batch = orientation_prob[None].expand(C_s,-1,-1)

    #    for n_idx in range(N_s):
     #       batch = int_sample[n_idx*batch_size:(n_idx+1)*batch_size] 
        #    if n_idx == 0 or not self.N_s_same_graph:
        adj_matrix = self.sample_adj_matrix(edge_prob_batch, orientation_prob_batch, mirror_graphs, C_s, int_idx)   
        adj_matrices.append(adj_matrix)

        for c_idx, (C, start_idx) in enumerate(C_s_list):
            adj_matrix_expanded = adj_matrix[start_idx:start_idx+C,None].expand(-1,int_batch.shape[0],-1,-1).flatten(0,1)
            batch_exp = int_batch[None,:].expand(C,-1,-1).flatten(0,1)
            nll = self.evaluate_likelihoods(batch_exp, adj_matrix_expanded, int_idx, model)
            nll = nll.reshape(C, int_batch.shape[0], -1)
        
                
            try:
                logregret[c_idx] += nll.mean(dim=1)
            except:
                logregret.append(nll.mean(dim=1))

        adj_matrices = torch.cat(adj_matrices, dim=0)
        logregret = torch.cat(logregret, dim=0)

        return adj_matrices, logregret

        
        
    def sample_adj_matrix(self, edge_prob_batch, orientation_prob_batch, mirror_graphs, C_s, int_idx):
        sample_matrix = torch.bernoulli(edge_prob_batch * orientation_prob_batch)
        sample_matrix = sample_matrix * (1 - torch.eye(sample_matrix.shape[-1], device=sample_matrix.device)[None])
        if mirror_graphs:
            sample_matrix[C_s//2:] = sample_matrix[:C_s//2]
            sample_matrix[C_s//2:,int_idx] = 1 - sample_matrix[C_s//2:,int_idx]
            sample_matrix[:,int_idx,int_idx] = 0.
        return sample_matrix
    
    def _get_next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch
    
    @torch.no_grad()
    def evaluate_likelihoods(self, int_sample, adj_matrix, var_idx, model):
        model.eval()
        device = self.get_device(model)
        int_sample = int_sample.to(device)
        adj_matrix = adj_matrix.to(device)
		# Transpose for mask because adj[i,j] means that i->j
        mask_adj_matrix = adj_matrix.transpose(1,2)
        preds = model(int_sample, mask=mask_adj_matrix)
		
        preds = preds.flatten(0, 1)
        labels = int_sample.clone() 
        labels[:,var_idx] = -1 
        labels = labels.reshape(-1) 
        ll = F.cross_entropy(preds, labels, reduction='none', ignore_index=-1) 
        ll = ll.reshape(*int_sample.shape) 
        model.train() 
        return ll
    
    def get_device(self, model):
        if isinstance(model, nn.DataParallel):
            device = "cuda:0" 
        else:
            device = model.device
        return device

    def update(self, int_idx, adj_matrices, logregret):
        if self.gamma.grad is not None:
            self.gamma.grad.fill_(0)
        gamma_grads, theta_grads, debug = self.gradient_estimator(adj_matrices, logregret, int_idx)
        self.gamma.grad = gamma_grads
        self.theta.grad = theta_grads
            
        return debug["theta_mask"]
        
    @torch.no_grad()
    def gradient_estimator(self, adj_matrices, logregret, int_idx):
        batch_size = adj_matrices.shape[0]
        logregret = logregret.unsqueeze(dim=1)

        comp_probs = torch.sigmoid(self.theta)
        edge_probs = torch.sigmoid(self.gamma)
        
        
        num_pos = adj_matrices.sum(dim=0)
        num_neg = batch_size - num_pos
        mask = ((num_pos > 0) * (num_neg > 0)).float()
        pos_grads = (logregret * adj_matrices).sum(dim=0) / num_pos.clamp_(min=1e-5)
        neg_grads = (logregret * (1 - adj_matrices)).sum(dim=0) / num_neg.clamp_(min=1e-5)
        gamma_grads = mask * edge_probs * (1 - edge_probs) * comp_probs * (pos_grads - neg_grads + self.lambda_sparse)

        theta_inner_grads = edge_probs * (pos_grads - neg_grads)
        if self.theta_regularizer:
            theta_inner_grads += 0.5 * self.lambda_sparse * (edge_probs - edge_probs.T)
        theta_grads = mask * comp_probs * (1 - comp_probs) * theta_inner_grads
        theta_grads[:int_idx] = 0
        theta_grads[int_idx+1:] = 0
        theta_grads -= theta_grads.transpose(0, 1)

        # Creating a mask which theta's are actually updated
        theta_mask = torch.zeros_like(theta_grads)
        theta_mask[int_idx] = 1.
        theta_mask[:,int_idx] = 1.
        theta_mask[int_idx,int_idx] = 0

        debug = {
            "theta_grads": theta_grads,
            "comp_probs": comp_probs,
            "logregret": logregret,
            "theta_mask": theta_mask
        }
        
        return gamma_grads, theta_grads, debug
        