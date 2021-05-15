import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np 
import sys
sys.path.append("../")

from causal_discovery.graph_discovery import GraphDiscovery
from causal_discovery.graph_fitting import GraphFitting
from causal_discovery.multivariable_mlp import create_model
from DAG_generation.critics import SimpleRELAXCritic
from DAG_generation.utils import make_permutation_matrix, combine_order_and_adjmatrix
from DAG_generation.estimators import to_z, to_b
from DAG_generation.graph_scoring import GraphScoringDAG
from DAG_generation.graph_update import GraphUpdateDAG
from experiments.utils import track


class GraphDiscoveryDAG(GraphDiscovery):

    def __init__(self, *args,
                       lr_permut=5e-3,
                       **kwargs):
        super().__init__(*args,
                         **kwargs,
                         fittingClass=GraphFitting,
                         scoringClass=GraphScoringDAG,
                         updateClass=GraphUpdateDAG,
                         lr_permut=lr_permut)


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
        for _ in track(range(self.gamma_iters), leave=False, desc="Gamma update loop"):
            self.gamma_optimizer.zero_grad()
            self.pl_optimizer.zero_grad()
            gammagrad, logregret, gammamask, theta_z, theta_b, var_idx = self.scoringModule.score(self.gamma, self.pl_thetas)
            self.updateModule.update(gammagrad, logregret, gammamask, self.gamma, var_idx,
                                     theta_z, theta_b, self.pl_thetas, self.pl_critic)
            self.gamma_optimizer.step()
            self.pl_optimizer.step()


    def _get_adjmatrix(self):
        return self.gamma


    @torch.no_grad()
    def sample_func(self, sample_matrix, batch_size):
        A = sample_matrix[None].expand(batch_size, -1, -1)
        A = torch.bernoulli(A)
        z = to_z(self.pl_thetas[None].expand(batch_size, -1), req_grads=False)
        b = to_b(z)
        A = combine_order_and_adjmatrix(A, b)
        return A


    def get_binary_adjmatrix(self):
        order = torch.argsort(-self.pl_thetas)
        A = (self._get_adjmatrix() > 0)
        A = combine_order_and_adjmatrix(A, order)
        return (A == 1)


    def init_gamma_params(self, num_vars, lr_gamma, betas_gamma, lr_permut, **kwargs):
        # We do not have to mask the triangular matrix because we train both directions
        self.gamma = nn.Parameter(torch.zeros(num_vars, num_vars)) 
        self.gamma.data[torch.arange(num_vars), torch.arange(num_vars)] = -9e15
        if betas_gamma[1] > 0:
            self.gamma_optimizer = torch.optim.Adam([self.gamma], lr=lr_gamma, betas=betas_gamma)
        else:
            self.gamma_optimizer = torch.optim.SGD([self.gamma], lr=lr_gamma, momentum=betas_gamma[0])
        self.pl_thetas = nn.Parameter(torch.zeros(num_vars))
        self.pl_critic = SimpleRELAXCritic(num_vars, 32) # Loss function f has to set every iteration, data-depending
        self.pl_optimizer = torch.optim.Adam([self.pl_thetas] + list(self.pl_critic.parameters()), lr=lr_gamma)


    @torch.no_grad()
    def print_gamma_statistics(self, print_gamma=False):
        super().print_gamma_statistics(print_gamma=print_gamma)
        m = self.get_metrics()
        print("Theta - Direction accuracy: %4.2f%% (TP=%i,FN=%i)" % (m["order"]["accuracy"] * 100.0, m["order"]["TP"], m["order"]["FN"]))


    @torch.no_grad()
    def get_metrics(self):
        metrics = super().get_metrics()
        order = torch.argsort(-self.pl_thetas)
        true_adj_matrix = self.true_adj_matrix.float()
        A = combine_order_and_adjmatrix(true_adj_matrix, order)
        TP = torch.logical_and(true_adj_matrix == 1, A == 1).float().sum().item()
        FN = torch.logical_and(true_adj_matrix == 1, A == 0).float().sum().item()
        acc = TP / max(1e-5, TP + FN)

        metrics["order"] = {
            "TP": int(TP),
            "FN": int(FN),
            "accuracy": acc
        }
        return metrics


    def to(self, device):
        self.fittingModule.model.to(device)
        self.gamma.to(device)
        self.pl_thetas.to(device)
        self.pl_critic.to(device)