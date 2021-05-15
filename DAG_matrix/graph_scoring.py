import torch
import torch.utils.data as data 
import torch.nn.functional as F
import numpy as np 
import math
import random
import time
import sys
sys.path.append("../")

from causal_discovery.graph_scoring import GraphScoring


class GraphScoringMatrix(GraphScoring):


    def __init__(self, *args, max_graph_stacking=1000, pretrain_C_s=-1, pretrain_N_s=-1, N_s_same_graph=False, **kwargs):
        kwargs["guide_inter"] = False
        super().__init__(*args, **kwargs)
        self.max_graph_stacking = max_graph_stacking
        self.N_s_same_graph = N_s_same_graph
        self.pretrain_C_s = self.C_s // 5 if pretrain_C_s <= 0 else pretrain_C_s
        self.pretrain_N_s = self.N_s if pretrain_N_s <= 0 else pretrain_N_s
        if self.graph.num_vars >= 100:
            print("Sampling interventional data...")
            start_time = time.time()
            self.dataset = InterventionalDataset(self.graph, dataset_size=self.batch_size*self.N_s*32, batch_size=self.batch_size)
            print("Done in %4.2fs" % (time.time() - start_time))


    def score(self, gamma, theta_matrix, var_idx=-1, return_debug=False, only_theta=False):
        if not only_theta:
            return self._perform_scoring(gamma, theta_matrix, N_s=self.N_s, C_s=self.C_s, batch_size=self.batch_size,
                                         var_idx=var_idx, return_debug=return_debug, mirror_graphs=False)
        else:
            return self._perform_scoring(gamma, theta_matrix, N_s=self.pretrain_N_s, C_s=self.pretrain_C_s, batch_size=self.batch_size,
                                         var_idx=var_idx, return_debug=return_debug, mirror_graphs=True)

    @torch.no_grad()
    def _perform_scoring(self, gamma, theta_matrix, N_s, C_s, batch_size, 
                               var_idx=-1, return_debug=False, mirror_graphs=False):
        if mirror_graphs:
            assert C_s % 2 == 0, "Number of graphs must be divisible by two for mirroring" 
        device = self.get_device()
        start_time = time.time()
        if hasattr(self, "dataset"):
            var_idx = self.sample_next_var_idx()
            int_sample = torch.cat([self.dataset.get_batch(var_idx) for _ in range(N_s)], dim=0).to(device)
        else:
            intervention_dict, var_idx = self.sample_intervention(self.graph, 
                                                                  dataset_size=N_s*batch_size,
                                                                  gamma=gamma,
                                                                  var_idx=var_idx)
            int_sample = self.graph.sample(interventions=intervention_dict, 
                                           batch_size=N_s*batch_size, 
                                           as_array=True)
            int_sample = torch.from_numpy(int_sample).long().to(device)
        t = time.time() - start_time
        # print("Sampling completed in %4.2fs" % (t))
        start_time = time.time()
        
        adj_matrices = []
        logregret = []
        debug = {}

        C_s_list = [min(self.max_graph_stacking, C_s-i*self.max_graph_stacking) for i in range(math.ceil(C_s * 1.0 / self.max_graph_stacking))]
        C_s_list = [(C_s_list[i],sum(C_s_list[:i])) for i in range(len(C_s_list))]

        edge_prob = torch.sigmoid(gamma).detach()
        orientation_prob = torch.sigmoid(theta_matrix).detach()
        edge_prob_batch = edge_prob[None].expand(C_s,-1,-1)
        orientation_prob_batch = orientation_prob[None].expand(C_s,-1,-1)

        
        def sample_adj_matrix():
            sample_matrix = torch.bernoulli(edge_prob_batch * orientation_prob_batch)
            sample_matrix = sample_matrix * (1 - torch.eye(sample_matrix.shape[-1], device=sample_matrix.device)[None])
            if mirror_graphs:
                sample_matrix[C_s//2:] = sample_matrix[:C_s//2]
                sample_matrix[C_s//2:,var_idx] = 1 - sample_matrix[C_s//2:,var_idx]
                sample_matrix[:,var_idx,var_idx] = 0.
            return sample_matrix

        
        for n_idx in range(N_s):
            batch = int_sample[n_idx*batch_size:(n_idx+1)*batch_size] 
            if n_idx == 0 or not self.N_s_same_graph:
                adj_matrix = sample_adj_matrix()   
                adj_matrices.append(adj_matrix)

            for c_idx, (C, start_idx) in enumerate(C_s_list):
                adj_matrix_expanded = adj_matrix[start_idx:start_idx+C,None].expand(-1,batch_size,-1,-1).flatten(0,1)
                batch_exp = batch[None,:].expand(C,-1,-1).flatten(0,1)
                nll = self.evaluate_likelihoods(batch_exp, adj_matrix_expanded, var_idx)
                nll = nll.reshape(C, batch_size, -1)

                if n_idx == 0 or not self.N_s_same_graph:
                    logregret.append(nll.mean(dim=1))
                else:
                    logregret[c_idx] += nll.mean(dim=1)

        adj_matrices = torch.cat(adj_matrices, dim=0)
        logregret = torch.cat(logregret, dim=0)
        if self.N_s_same_graph:
            logregret /= N_s

        t = time.time() - start_time
        # print("Completed rest of the loop in %4.2fs" % t)

        if not return_debug:
            return adj_matrices, logregret, var_idx
        else:
            return adj_matrices, logregret, var_idx, debug



class InterventionalDataset:

    def __init__(self, graph, dataset_size, batch_size, num_stacks=50):
        self.graph = graph 
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.data_loaders = {}
        self.data_iter = {}
        intervention_list = []
        for var_idx in range(self.graph.num_vars):
            var = self.graph.variables[var_idx]
            values = np.random.randint(var.prob_dist.num_categs, size=(dataset_size,))
            intervention_list.append((var_idx, var, values))
            if len(intervention_list) >= num_stacks:
                self._add_vars(intervention_list)
                intervention_list = []
        if len(intervention_list) > 0:
            self._add_vars(intervention_list)

    def _add_vars(self, intervention_list):
        num_vars = len(intervention_list)
        intervention_dict = {}
        for i, (var_idx, var, values) in enumerate(intervention_list):
            v_array = -np.ones((num_vars, self.dataset_size), dtype=np.int32)
            v_array[i] = values
            v_array = np.reshape(v_array, (-1,))
            intervention_dict[var.name] = v_array
        int_sample = self.graph.sample(interventions=intervention_dict, 
                                       batch_size=self.dataset_size*num_vars, 
                                       as_array=True)
        int_sample = torch.from_numpy(int_sample).long().reshape(num_vars, self.dataset_size, int_sample.shape[-1])
        for i, (var_idx, var, values) in enumerate(intervention_list):
            dataset = data.TensorDataset(int_sample[i])
            self.data_loaders[var_idx] = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False)
            self.data_iter[var_idx] = iter(self.data_loaders[var_idx])


    def get_batch(self, var_idx):
        try:
            batch = next(self.data_iter[var_idx])
        except StopIteration:
            self.data_iter[var_idx] = iter(self.data_loaders[var_idx])
            batch = next(self.data_iter[var_idx])
        return batch[0]