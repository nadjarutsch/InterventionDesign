import torch
import torch.utils.data as data 
import torch.nn.functional as F
import numpy as np 
import random
import sys
sys.path.append("../")

from causal_discovery.graph_scoring import GraphScoring
from DAG_generation.estimators import to_z, to_b
from DAG_generation.utils import combine_order_and_adjmatrix


class GraphScoringDAG(GraphScoring):


    def __init__(self, *args, **kwargs):
        kwargs["guide_inter"] = False
        super().__init__(*args, **kwargs)


    @torch.no_grad()
    def score(self, gamma, log_theta, var_idx=-1, return_debug=False):
        intervention_dict, var_idx = self.sample_intervention(self.graph, 
                                                              dataset_size=self.N_s*self.batch_size,
                                                              gamma=gamma,
                                                              var_idx=var_idx)
        int_sample = self.graph.sample(interventions=intervention_dict, 
                                       batch_size=self.N_s*self.batch_size, 
                                       as_array=True)
        int_sample = torch.from_numpy(int_sample)

        gammagrad = []
        logregret = []
        gammamask = []
        theta_z = []
        theta_b = []
        debug = {}

        sample_matrix = torch.sigmoid(gamma).detach()
        for n_idx in range(self.N_s):
            batch = torch.LongTensor(int_sample[n_idx*self.batch_size:(n_idx+1)*self.batch_size])
            debug[n_idx] = list()
            for c_idx in range(self.C_s):
                debug[n_idx].append(dict())
                adj_matrix = torch.bernoulli(sample_matrix)
                z = to_z(log_theta, req_grads=True)
                b = to_b(z)
                adj_matrix, adj_mask = combine_order_and_adjmatrix(adj_matrix, b, return_mask=True)
                debug[n_idx][-1]["adj_matrix"] = adj_matrix
                debug[n_idx][-1]["adj_mask"] = adj_mask
                adj_matrix = adj_matrix[None].expand(self.batch_size, -1, -1)
                nll = self.evaluate_likelihoods(batch, adj_matrix, var_idx)
                debug[n_idx][-1]["nll"] = nll
                gammagrad.append(sample_matrix - adj_matrix[0])
                logregret.append(nll.mean(dim=0))
                gammamask.append(adj_mask)
                theta_z.append(z)
                theta_b.append(b)

                # print("Sampled adj matrix:\n", adj_matrix[0])
                # print("NLL", nll.mean(dim=0))
        gammagrad = torch.stack(gammagrad, dim=0)
        logregret = torch.stack(logregret, dim=0)
        gammamask = torch.stack(gammamask, dim=0)
        theta_z = torch.stack(theta_z, dim=0)
        theta_b = torch.stack(theta_b, dim=0)

        if not return_debug:
            return gammagrad, logregret, gammamask, theta_z, theta_b, var_idx
        else:
            return gammagrad, logregret, gammamask, theta_z, theta_b, var_idx, debug