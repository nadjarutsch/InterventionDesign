from enco_model import *
from env import CausalEnv
from heuristics import choose_intervention
from utils import track
from metrics import *

import argparse 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from collections import defaultdict





class OptimizerTemplate:

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    @torch.no_grad()
    def step(self):
        ## Apply update step to all parameters
        for p in self.params:
            if p.grad is None: # We skip parameters without any gradients
                continue
            self.update_param(p)

    def update_param(self, p):
        # To be implemented in optimizer-specific classes
        raise NotImplementedError


class AdamTheta(OptimizerTemplate):

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.param_step = torch.zeros_like(self.params.data) # Remembers "t" for each parameter for bias correction
        self.param_momentum = torch.zeros_like(self.params.data)
        self.param_2nd_momentum = torch.zeros_like(self.params.data)

    def zero_grad(self):
        ## Set gradients of all parameters to zero
        if self.params.grad is not None:
            self.params.grad.detach_() # For second-order optimizers important
            self.params.grad.zero_()

    @torch.no_grad()
    def step(self, mask):
        if self.params.grad is None:
            return 

        self.param_step.add_(mask)

        new_momentum = (1 - self.beta1) * self.params.grad + self.beta1 * self.param_momentum
        new_2nd_momentum = (1 - self.beta2) * (self.params.grad)**2 + self.beta2 * self.param_2nd_momentum
        self.param_momentum = torch.where(mask == 1.0, new_momentum, self.param_momentum)
        self.param_2nd_momentum = torch.where(mask == 1.0, new_2nd_momentum, self.param_2nd_momentum)

        bias_correction_1 = 1 - self.beta1 ** self.param_step
        bias_correction_2 = 1 - self.beta2 ** self.param_step
        bias_correction_1.masked_fill_(bias_correction_1 == 0.0, 1.0)
        bias_correction_2.masked_fill_(bias_correction_2 == 0.0, 1.0)

        p_2nd_mom = self.param_2nd_momentum / bias_correction_2
        p_mom = self.param_momentum / bias_correction_1
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom
        p_update = mask * p_update

        self.params.add_(p_update)

    @torch.no_grad()
    def to(self, device):
        self.param_step = self.param_step.to(device)
        self.param_momentum = self.param_momentum.to(device)
        self.param_2nd_momentum = self.param_2nd_momentum.to(device)



def distribution_fitting(args: argparse.Namespace,
                         model: MultivarMLP,
                         loss_module: nn.CrossEntropyLoss,
                         optimizer: torch.optim.Adam,
                         adj_matrix: AdjacencyMatrix,
                         dataloader: DataLoader) -> float:
    """Fit the multivariable MLP to observational data, given the predicted
    adjacency matrix.
    
    Args:
        args: Object from the argument parser that defines various settings of
            the causal structure and discovery process.
        distributionFitting: Module used for fitting the MLP to observational 
            data and the predicted adjacency matrix.
        gamma: Matrix of gamma values (determining edge probabilities).
        theta: Matrix of theta values (determining edge directions).
        dataloader: Dataloader with observational data from the joint distri-
            bution.
    
    Returns:
        distributionFitting: Fitting module with updated MLP.
        avg_loss: Average loss in one fitting epoch.
    """
    sample_matrix = torch.sigmoid(adj_matrix.gamma.detach())
    sfunc = lambda batch_size: sample_func(sample_matrix=sample_matrix, 
                                           theta=adj_matrix.theta,
                                           batch_size=args.obs_batch_size) 
        
    avg_loss = 0.0
    t = track(range(args.obs_epochs), leave=False, desc="Model update loop")
    for _ in t:
        batch = next(iter(dataloader))
        adj_matrices = sfunc(batch_size=batch.shape[0])

        model.train()
        device = get_device(model)
        batch = batch.to(device)
        adj_matrices = adj_matrices.to(device)
		# Transpose for mask because adj[i,j] means that i->j
        mask_adj_matrices = adj_matrices.transpose(1,2)
        preds = model(batch, mask=mask_adj_matrices)
        loss_module = loss_module.to(device)
        loss = loss_module(preds.flatten(0,-2), batch.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if hasattr(t, "set_description"):
            t.set_description("Model update loop, loss: %4.2f" % loss)

    avg_loss /= args.obs_epochs
    
    return avg_loss


def sample_func(sample_matrix, theta, batch_size):
    
    A = sample_matrix[None].expand(batch_size, -1, -1)
    A = torch.bernoulli(A)
    order_p = torch.sigmoid(theta) * (1 - torch.eye(theta.shape[0], device=theta.device))
    order_p = order_p[None].expand(batch_size, -1, -1)
    order_mask = torch.bernoulli(order_p)

    A = A * order_mask

    return A 



def graph_fitting(args: argparse.Namespace,
                  adj_matrix: AdjacencyMatrix,
                  gamma_optimizer: torch.optim.Adam or torch.optim.SGD,
                  theta_optimizer: AdamTheta,
                  model: MultivarMLP,
                  env: CausalEnv,
                  epoch: int,
                  logger: Logger):
    """Fit the adjacency matrix to interventional data, given the learned 
    multivariable MLP.
    
    Args:
        args: Object from the argument parser that defines various settings of
            the causal structure and discovery process.
        updateModule: Module used for fitting the graph to interventional data.
        model: Multivariable MLP (modelling the conditional distributions).
        int_idx: Index of the intervention variable.
        int_dataloader: Dataloader with interventional data.
        epoch: Current epoch, used for theta pretraining threshold.
    
    Returns:
        updateModule: Graph fitting module with updated adjacency matrix.
    """
    logger.stats = defaultdict(int) 
    
    for i in track(range(args.int_epochs), leave=False, desc="Gamma and theta update loop"):
        # perform intervention and update parameters based on interventional data
        if epoch == 0 and i == 0: # always start with the same variable
            int_idx = 0
        else:
            true_adj_matrix = torch.from_numpy(env.dag.adj_matrix).float().to(adj_matrix.gamma.device)
            int_idx = choose_intervention(args, i, adj_matrix=adj_matrix, true_adj=true_adj_matrix)
        logger.stats[int_idx] += 1 
        int_data, reward, info = env.step(int_idx, args.int_batch_size) 
        int_dataloader = DataLoader(int_data, batch_size=args.int_batch_size, shuffle=True, drop_last=True)
        
        gamma_optimizer.zero_grad()
        theta_optimizer.zero_grad()
        batch = next(iter(int_dataloader))
        adj_matrices, logregret = score(batch, int_idx, adj_matrix, model)
        theta_mask = update(args, int_idx, adj_matrix, adj_matrices, logregret) 

        gamma_optimizer.step()
        theta_optimizer.step(theta_mask)

 
 
def score(int_batch, 
          int_idx,
          adj_matrix,
          model, 
          mirror_graphs=False, 
          C_s=20,
          max_graph_stacking=1000):

        adj_matrices = []
        logregret = []

        C_s_list = [min(max_graph_stacking, C_s-i*max_graph_stacking) for i in range(math.ceil(C_s * 1.0 / max_graph_stacking))]
        C_s_list = [(C_s_list[i],sum(C_s_list[:i])) for i in range(len(C_s_list))]

        edge_prob = adj_matrix.edge_probs().detach()
        edge_prob_batch = edge_prob[None].expand(C_s,-1,-1)
        sample_matrix = torch.bernoulli(edge_prob_batch)
        sample_matrix = sample_matrix * (1 - torch.eye(sample_matrix.shape[-1], device=sample_matrix.device)[None])
 
        adj_matrices.append(sample_matrix)

        for c_idx, (C, start_idx) in enumerate(C_s_list):
            adj_matrix_expanded = sample_matrix[start_idx:start_idx+C,None].expand(-1,int_batch.shape[0],-1,-1).flatten(0,1)
            batch_exp = int_batch[None,:].expand(C,-1,-1).flatten(0,1)
            nll = evaluate_likelihoods(batch_exp, adj_matrix_expanded, int_idx, model)
            nll = nll.reshape(C, int_batch.shape[0], -1)
        
                
            try:
                logregret[c_idx] += nll.mean(dim=1)
            except:
                logregret.append(nll.mean(dim=1))

        adj_matrices = torch.cat(adj_matrices, dim=0)
        logregret = torch.cat(logregret, dim=0)

        return adj_matrices, logregret
  
    
def update(args, int_idx, adj_matrix, adj_matrices, logregret):
        if adj_matrix.gamma.grad is not None:
            adj_matrix.gamma.grad.fill_(0)
        gamma_grads, theta_grads, debug = gradient_estimator(args, adj_matrix, adj_matrices, logregret, int_idx)
        adj_matrix.gamma.grad = gamma_grads
        adj_matrix.theta.grad = theta_grads
            
        return debug["theta_mask"]
        
@torch.no_grad()
def gradient_estimator(args, adj_matrix, adj_matrices, logregret, int_idx):
    batch_size = adj_matrices.shape[0]
    logregret = logregret.unsqueeze(dim=1)

    comp_probs = torch.sigmoid(adj_matrix.theta)
    edge_probs = torch.sigmoid(adj_matrix.gamma)
    
    
    num_pos = adj_matrices.sum(dim=0)
    num_neg = batch_size - num_pos
    mask = ((num_pos > 0) * (num_neg > 0)).float()
    pos_grads = (logregret * adj_matrices).sum(dim=0) / num_pos.clamp_(min=1e-5)
    neg_grads = (logregret * (1 - adj_matrices)).sum(dim=0) / num_neg.clamp_(min=1e-5)
    gamma_grads = mask * edge_probs * (1 - edge_probs) * comp_probs * (pos_grads - neg_grads + args.lambda_sparse)

    theta_inner_grads = edge_probs * (pos_grads - neg_grads)
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
 
 
@torch.no_grad()
def evaluate_likelihoods(int_sample, adj_matrix, var_idx, model):
    model.eval()
    device = get_device(model)
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
 
 
def get_device(model):
    if isinstance(model, nn.DataParallel):
        device = "cuda:0" 
    else:
        device = model.device
    return device 
 
 

