import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np 
import time
import sys
sys.path.append("../")

from causal_discovery.graph_discovery import GraphDiscovery
from causal_discovery.graph_fitting import GraphFitting
from causal_discovery.multivariable_mlp import create_model
from DAG_generation.utils import make_permutation_matrix, combine_order_and_adjmatrix
from DAG_matrix.graph_scoring import GraphScoringMatrix
from DAG_matrix.graph_update import GraphUpdateMatrix
from DAG_matrix.utils import matrix_to_order, update_stats, get_final_stats
from DAG_matrix.adam_theta import AdamTheta
from experiments.utils import track


class GraphDiscoveryMatrix(GraphDiscovery):

    def __init__(self, *args,
                       lr_permut=5e-3,
                       C_s=200,
                       max_graph_stacking=200,
                       order_grads_opt='plain',
                       edge_grads_opt='plain',
                       theta_regularizer=False,
                       N_s_same_graph=False,
                       warmup_steps=100,
                       betas_gamma=(0.9,0.999),
                       betas_theta=(0.9,0.999),
                       use_adam_theta=False,
                       theta_pretraining=0,
                       theta_alternate=False,
                       pretrain_C_s=-1,
                       pretrain_N_s=-1,
                       pretrain_iters=-1,
                       **kwargs):
        super().__init__(*args,
                         **kwargs,
                         fittingClass=GraphFitting,
                         scoringClass=lambda *args, **kwargs: GraphScoringMatrix(*args, **kwargs, 
                                                                                 N_s_same_graph=N_s_same_graph, 
                                                                                 max_graph_stacking=max_graph_stacking,
                                                                                 pretrain_C_s=pretrain_C_s,
                                                                                 pretrain_N_s=pretrain_N_s),
                         updateClass=lambda *args, **kwargs: GraphUpdateMatrix(*args, **kwargs, 
                                                                               order_grads_opt=order_grads_opt, 
                                                                               edge_grads_opt=edge_grads_opt,
                                                                               theta_regularizer=theta_regularizer),
                         lr_permut=lr_permut,
                         C_s=C_s,
                         warmup_steps=warmup_steps,
                         betas_theta=betas_theta,
                         use_adam_theta=use_adam_theta)
        self.theta_pretraining = theta_pretraining
        self.theta_alternate = theta_alternate
        self.pretrain_iters = 10*self.gamma_iters if pretrain_iters <= 0 else pretrain_iters
        self.metric_log = []
        self.theta_matrix_grad_log = []
        self.theta_matrix_log = []
        self.gamma_grad_log = []
        self.gamma_log = []
        self.gamma_grad_stats = {}
        self.theta_grad_stats = {}


    def model_fitting_step(self):
        sample_matrix = torch.sigmoid(self._get_adjmatrix().detach())
        sfunc = lambda batch_size: self.sample_func(sample_matrix=sample_matrix, 
                                                    batch_size=batch_size)
        
        avg_loss = 0.0
        t = track(range(self.model_iters), leave=False, desc="Model update loop")
        for _ in t:
            loss = self.fittingModule.fit_step(sample_func=sfunc)
            avg_loss += loss
            if hasattr(t, "set_description"):
                t.set_description("Model update loop, loss: %4.2f" % loss)
        avg_loss /= self.model_iters


    def gamma_update_step(self):
        only_theta = (self.epoch < self.theta_pretraining) or (self.theta_alternate and self.epoch % 2 == 0)
        iters = self.gamma_iters if not only_theta else self.pretrain_iters
        for _ in track(range(iters), leave=False, desc="Gamma update loop"):
            self.gamma_optimizer.zero_grad()
            self.theta_optimizer.zero_grad()
            adj_matrices, logregret, var_idx = self.scoringModule.score(self.gamma, self.theta_matrix, 
                                                                        only_theta=only_theta)
            theta_mask = self.updateModule.update(adj_matrices, logregret, self.gamma, self.theta_matrix, var_idx)
            if not only_theta:
                self.gamma_optimizer.step()
            # self.gamma_scheduler.step()
            if isinstance(self.theta_optimizer, AdamTheta):
                self.theta_optimizer.step(theta_mask)
            else:
                self.theta_optimizer.step()
            # self.theta_scheduler.step()
            self.log_params()
        self.log_params(end_epoch=True)


    def log_params(self, end_epoch=False, max_vars=30):
        tensors = [self.theta_matrix.data, self.theta_matrix.grad, self.gamma.data, self.gamma.grad]
        tmat, tmat_grad, gam, gam_grad = [t.detach().clone().cpu() for t in tensors]
        if not end_epoch:
            if self.num_vars < max_vars:
                self.theta_matrix_log.append(tmat)
                self.theta_matrix_grad_log.append(tmat_grad)
                self.gamma_log.append(gam)
                self.gamma_grad_log.append(gam_grad)
            else:
                update_stats(self.gamma_grad_stats, gam_grad)
                update_stats(self.theta_grad_stats, tmat_grad)
        elif end_epoch and self.num_vars >= max_vars:
            self.theta_matrix_log.append(tmat)
            self.gamma_log.append(gam)
            self.theta_matrix_grad_log.append(get_final_stats(self.theta_grad_stats))
            self.gamma_grad_log.append(get_final_stats(self.gamma_grad_stats))
            self.theta_grad_stats = {}
            self.gamma_grad_stats = {}


    def _get_adjmatrix(self):
        return self.gamma


    @torch.no_grad()
    def sample_func(self, sample_matrix, batch_size):
        A = sample_matrix[None].expand(batch_size, -1, -1)
        A = torch.bernoulli(A)
        order_p = torch.sigmoid(self.theta_matrix) * (1 - torch.eye(self.theta_matrix.shape[0], device=self.theta_matrix.device))
        order_p = order_p[None].expand(batch_size, -1, -1)
        order_mask = torch.bernoulli(order_p)

        A = A * order_mask

        return A


    def get_binary_adjmatrix(self):
        order = matrix_to_order(self.theta_matrix)
        A = (self._get_adjmatrix() > 0.0) * (self.theta_matrix > 0.0)
        # A = combine_order_and_adjmatrix(A, order)
        return (A == 1).cpu()


    def init_gamma_params(self, num_vars, lr_gamma, betas_gamma, lr_permut, betas_theta, warmup_steps, use_adam_theta, **kwargs):
        # We do not have to mask the triangular matrix because we train both directions
        self.gamma = nn.Parameter(torch.zeros(num_vars, num_vars)) 
        self.gamma.data[torch.arange(num_vars), torch.arange(num_vars)] = -9e15
        if betas_gamma[1] > 0:
            self.gamma_optimizer = torch.optim.Adam([self.gamma], lr=lr_gamma, betas=betas_gamma)
        else:
            self.gamma_optimizer = torch.optim.SGD([self.gamma], lr=lr_gamma, momentum=betas_gamma[0])
        # self.gamma_scheduler = WarmupScheduler(self.gamma_optimizer, warmup_steps)

        self.theta_matrix = nn.Parameter(torch.zeros(num_vars,num_vars))
        if use_adam_theta:
            self.theta_optimizer = AdamTheta(self.theta_matrix, lr=lr_permut, beta1=betas_theta[0], beta2=betas_theta[1])
        elif betas_theta[1] > 0:
            self.theta_optimizer = torch.optim.Adam([self.theta_matrix], lr=lr_permut, betas=betas_theta)
        else:
            self.theta_optimizer = torch.optim.SGD([self.theta_matrix], lr=lr_permut, momentum=betas_theta[0])
        # self.theta_scheduler = WarmupScheduler(self.pl_optimizer, warmup_steps)


    @torch.no_grad()
    def print_gamma_statistics(self, print_gamma=False, epoch=-1):
        super().print_gamma_statistics(print_gamma=print_gamma, epoch=epoch)
        m = self.get_metrics()
        print("Theta - Direction accuracy: %4.2f%% (TP=%i,FN=%i) / Soft accuracy: %4.2f%% (TP=%i,FN=%i)" % (m["order"]["accuracy"] * 100.0, m["order"]["TP"], m["order"]["FN"], m["order"]["soft_acc"]*100.0, m["order"]["soft_TP"], m["order"]["soft_FN"]))
        self.metric_log.append(m)

        if self.num_vars >= 100:
            print("-> Iteration time: %imin %is" % (int(self.iter_time)//60, int(self.iter_time)%60))
            print("-> Fitting time: %imin %is" % (int(self.fit_time)//60, int(self.fit_time)%60))
            gpu_mem = torch.cuda.max_memory_allocated(device="cuda:0")/1.0e9 if torch.cuda.is_available() else -1
            print("-> Used GPU memory: %4.2fGB" % (gpu_mem))


    @torch.no_grad()
    def get_metrics(self):
        metrics = super().get_metrics()
        order = matrix_to_order(self.theta_matrix)
        true_adj_matrix = self.true_adj_matrix.float().to(self.theta_matrix.device)
        A = combine_order_and_adjmatrix(true_adj_matrix, order)
        TP = torch.logical_and(true_adj_matrix == 1, A == 1).float().sum().item()
        FN = torch.logical_and(true_adj_matrix == 1, A == 0).float().sum().item()
        acc = TP / max(1e-5, TP + FN)

        soft_TP = torch.logical_and(true_adj_matrix == 1, self.theta_matrix > 0.0).float().sum().item()
        soft_FN = torch.logical_and(true_adj_matrix == 1, self.theta_matrix <= 0.0).float().sum().item()
        soft_acc = soft_TP / max(1e-5, soft_TP + soft_FN)

        metrics["order"] = {
            "TP": int(TP),
            "FN": int(FN),
            "accuracy": acc,
            "soft_TP": int(soft_TP),
            "soft_FN": int(soft_FN),
            "soft_acc": soft_acc
        }
        metrics["time_stamp"] = time.time()
        return metrics


    def to(self, device):
        self.fittingModule.model.to(device)
        self.gamma.data = self.gamma.data.to(device)
        self.theta_matrix.data = self.theta_matrix.data.to(device)
        if hasattr(self.theta_optimizer, "to"):
            self.theta_optimizer.to(device)


    def get_state_dict(self):
        state_dict = {
            "gamma": self.gamma.data.detach(),
            "theta_matrix": self.theta_matrix.data.detach(),
            "model": self.fittingModule.model.state_dict()
        }
        return state_dict


    def load_state_dict(self, state_dict):
        self.gamma.data = state_dict["gamma"]
        self.theta_matrix.data = state_dict["theta_matrix"]
        self.fittingModule.model.load_state_dict(state_dict["model"])