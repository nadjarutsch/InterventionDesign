from enco_model import *
from env import CausalEnv
from heuristics import choose_intervention
from utils import track
from metrics import *
from policy import MLPolicy

import argparse 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from collections import defaultdict





class OptimizerTemplate:

    def __init__(self, params, lr):
        self.params = params
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
        
    def zero_grad(self):
        # Set gradients of all parameters to zero
        if self.params.grad is not None:
            self.params.grad.detach_()
            self.params.grad.zero_()


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
        
        
class AdamGamma(OptimizerTemplate):

    def __init__(self, params, lr, beta1=0.9, beta2=0.9, eps=1e-8):
        """
        Adam optimizer for the gamma parameters when latent confounders should be detected. 
        The difference to standard Adam is that we track the gradients and first-order 
        momentum parameter for observational and interventional data separately. After
        training, the latent confounder scores can be calculated from the aggregated 
        gradients. The difference of gamma optimized via this optimizer vs standard Adam
        is neglectable, and did not show any noticable differences in performance. 

        Parameters
        ----------
        params : nn.Parameter / torch.Tensor with grads
                 The parameters that should be optimized, here representing gamma of ENCO.
        lr : float
             Basic learning rate of Adam.
        beta1 : float
                beta-1 hyperparameter of Adam.
        beta2 : float
                beta-2 hyperparameter of Adam.
        eps : float
              Epsilon hyperparameter of Adam.
        """
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.param_step = torch.zeros(self.params.data.shape + (2,), device=self.params.device)
        self.param_momentum = torch.zeros(self.params.data.shape + (2,), device=self.params.device)
        self.param_2nd_momentum = torch.zeros_like(self.params.data)  # Adaptive lr needs to shared of obs and int data
        self.updates = torch.zeros(self.params.data.shape + (2,), device=self.params.device)

    @torch.no_grad()
    def step(self, var_idx):
        """
        Standard Adam update step, except that it tracks the gradients of the variable 'var_idx'
        separately from all other variables.

        Parameters
        ----------
        var_idx : int
                  Index of the variable on which an intervention has been performed. The input 
                  should be negative in case no intervention had been performed.
        """
        if self.params.grad is None:
            return

        mask = torch.ones_like(self.params.data)
        mask_obs_int = torch.ones_like(self.param_step)
        if var_idx >= 0:
            mask[:, var_idx] = 0.0
            mask_obs_int[var_idx, :, 0] = 0.0
            mask_obs_int[..., 1] -= mask_obs_int[..., 0]
            mask_obs_int[:, var_idx, :] = 0.0

        self.param_step.add_(mask_obs_int)

        new_momentum = (1 - self.beta1) * self.params.grad[..., None] + self.beta1 * self.param_momentum
        new_2nd_momentum = (1 - self.beta2) * (self.params.grad)**2 + self.beta2 * self.param_2nd_momentum
        self.param_momentum = torch.where(mask_obs_int == 1.0, new_momentum, self.param_momentum)
        self.param_2nd_momentum = torch.where(mask == 1.0, new_2nd_momentum, self.param_2nd_momentum)

        bias_correction_1 = 1 - self.beta1 ** self.param_step
        bias_correction_2 = 1 - self.beta2 ** self.param_step.sum(dim=-1)
        bias_correction_1.masked_fill_(bias_correction_1 == 0.0, 1.0)
        bias_correction_2.masked_fill_(bias_correction_2 == 0.0, 1.0)

        p_2nd_mom = self.param_2nd_momentum / bias_correction_2
        p_mom = self.param_momentum / bias_correction_1
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr[..., None] * p_mom
        p_update = mask_obs_int * p_update

        self.params.add_(p_update.sum(dim=-1))
        self.updates.add_(p_update)

    @torch.no_grad()
    def to(self, device):
        self.param_step = self.param_step.to(device)
        self.param_momentum = self.param_momentum.to(device)
        self.param_2nd_momentum = self.param_2nd_momentum.to(device)
        self.updates = self.updates.to(device)



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
    sample_matrix = torch.sigmoid(adj_matrix.gamma.detach()) * torch.sigmoid(adj_matrix.theta.detach())
    sfunc = lambda batch_size: sample_func(sample_matrix=sample_matrix, 
                                           batch_size=args.obs_batch_size) 
        
    avg_loss = 0.0
    t = track(range(args.obs_epochs), leave=False, desc="Model update loop")
    data_iter = iter(dataloader)
    for _ in t:
        batch, data_iter = get_next_batch(dataloader, data_iter)
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



def get_next_batch(data_loader, data_iter):
    """
    Helper function for sampling batches one by one from the data loader.
    """
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        batch = next(data_iter)
    return batch, data_iter

@torch.no_grad()
def sample_func(sample_matrix, batch_size):
    """
    Samples a batch of adjacency matrices that are used for masking the inputs.
    """
    sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
    adj_matrices = torch.bernoulli(sample_matrix)
    # Mask diagonals
    adj_matrices[:, torch.arange(adj_matrices.shape[1]), torch.arange(adj_matrices.shape[2])] = 0
    return adj_matrices




def graph_fitting(args: argparse.Namespace,
                  adj_matrix: AdjacencyMatrix,
                  gamma_optimizer: torch.optim.Adam or torch.optim.SGD,
                  theta_optimizer: AdamTheta,
                  model: MultivarMLP,
                  env: CausalEnv,
                  epoch: int,
                  logger: Logger,
                  int_dists: defaultdict,
                  policy: MLPolicy):
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
    
    log_prob_sum = 0
    reward_sum = 0
    
    for i in track(range(args.int_epochs), leave=False, desc="Gamma and theta update loop"):
        # perform intervention and update parameters based on interventional data
        
        if args.learn_policy:
            state = (torch.sigmoid(adj_matrix.gamma.detach()), torch.sigmoid(adj_matrix.theta.detach()))
            int_idx, log_prob = policy.act(state)
        else:
            if epoch == 0 and i == 0: # always start with the same variable
                int_idx = 0
            else:
                true_adj_matrix = torch.from_numpy(env.dag.adj_matrix).float().to(adj_matrix.gamma.device)
                int_idx = choose_intervention(args, i, adj_matrix=adj_matrix, true_adj=true_adj_matrix)
            
        logger.stats[int_idx] += 1 
        int_data, reward, info = env.step(int_idx, args.int_batch_size, int_dists[int_idx]) 
        int_dataloader = DataLoader(int_data, batch_size=args.int_batch_size, shuffle=True, drop_last=True)
        batch = next(iter(int_dataloader))
        
        gamma_optimizer.zero_grad()
        theta_optimizer.zero_grad()
        adj_matrices, logregret = score(batch, int_idx, adj_matrix, model, K_s=args.K)
        theta_mask = update(args, int_idx, adj_matrix, adj_matrices, logregret) 
        gamma_optimizer.step(int_idx)
        theta_optimizer.step(theta_mask)
        
        log_prob_sum += log_prob
        reward_sum += reward

    return log_prob_sum, reward_sum
 

@torch.no_grad()
def score(int_batch, 
          int_idx,
          adj_matrix,
          model, 
          mirror_graphs=False, 
          K_s=100,
          max_graph_stacking=1000):

        adj_matrices = []
        logregret = []

        K_s_list = [min(max_graph_stacking, K_s-i*max_graph_stacking) for i in range(math.ceil(K_s * 1.0 / max_graph_stacking))]
        K_s_list = [(K_s_list[i],sum(K_s_list[:i])) for i in range(len(K_s_list))]

        edge_prob = adj_matrix.edge_probs().detach()
        edge_prob_batch = edge_prob[None].expand(K_s,-1,-1)
        sample_matrix = torch.bernoulli(edge_prob_batch)
        sample_matrix = sample_matrix * (1 - torch.eye(sample_matrix.shape[-1], device=sample_matrix.device)[None])
 
        adj_matrices.append(sample_matrix)

        for k_idx, (K, start_idx) in enumerate(K_s_list):
            adj_matrix_expanded = sample_matrix[start_idx:start_idx+K,None].expand(-1,int_batch.shape[0],-1,-1).flatten(0,1)
            batch_exp = int_batch[None,:].expand(K,-1,-1).flatten(0,1)
            nll = evaluate_likelihoods(batch_exp, adj_matrix_expanded, int_idx, model)
            nll = nll.reshape(K, int_batch.shape[0], -1)
        
                
            try:
                logregret[k_idx] += nll.mean(dim=1)
            except:
                logregret.append(nll.mean(dim=1))

        adj_matrices = torch.cat(adj_matrices, dim=0)
        logregret = torch.cat(logregret, dim=0)

        return adj_matrices, logregret
  

@torch.no_grad()   
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
    
    # Masking gamma for incoming edges to intervened variable
    gamma_grads[:, int_idx] = 0.
    gamma_grads[torch.arange(gamma_grads.shape[0]), torch.arange(gamma_grads.shape[1])] = 0.

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
 
 

