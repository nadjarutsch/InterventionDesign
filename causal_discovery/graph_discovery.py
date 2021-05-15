import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np 
import time
import sys
sys.path.append("../")

from causal_discovery.graph_fitting import GraphFitting
from causal_discovery.graph_scoring import GraphScoring
from causal_discovery.graph_update import GraphUpdate
from causal_discovery.multivariable_mlp import create_model
from experiments.utils import track



class GraphDiscovery(object):


    def __init__(self, graph, 
                       hidden_dims=[64], 
                       lr_model=2e-2, 
                       betas_model=(0.9,0.999),
                       lr_gamma=5e-3, 
                       betas_gamma=(0.1,0.1),
                       model_iters=10000,
                       gamma_iters=100,
                       N_s=10,
                       C_s=200,
                       lambda_sparse=0.001, 
                       lambda_DAG=0.05,
                       dataset_size=100000,
                       batch_size=128,
                       guide_inter=False,
                       share_embeds=False,
                       sparse_embeds=False,
                       data_parallel=False,
                       data_parallel_fitting=False,
                       actfn='leakyrelu',
                       fittingClass=GraphFitting,
                       scoringClass=GraphScoring,
                       updateClass=GraphUpdate,
                       **kwargs):
        num_vars = graph.num_vars
        num_categs = max([v.prob_dist.num_categs for v in graph.variables])
        # assert all([v.prob_dist.num_categs == num_categs for v in graph.variables]), \
        #      "Currently, only graphs with equal number of categories for all variables are supported."
        obs_dataset = ObservationalCategoricalData(graph, dataset_size=dataset_size)
        obs_data_loader = data.DataLoader(obs_dataset, batch_size=batch_size, 
                                          shuffle=True, drop_last=True)

        model = create_model(num_vars=num_vars, 
                             num_categs=num_categs, 
                             hidden_dims=hidden_dims, 
                             share_embeds=share_embeds,
                             actfn=actfn,
                             sparse_embeds=sparse_embeds)
        print(model)
        if data_parallel:
            print("Data parallel activated. Using %i GPUs" % torch.cuda.device_count())
            model_parallel = nn.DataParallel(model)
        else:
            model_parallel = model
        model_optimizer = torch.optim.Adam(model.parameters(), lr=lr_model, betas=betas_model)

        self.init_gamma_params(num_vars, lr_gamma, betas_gamma, **kwargs)
        self.num_vars = num_vars
        self.iter_time = -1

        self.fittingModule = fittingClass(model if not data_parallel_fitting else model_parallel, model_optimizer, obs_data_loader)
        self.scoringModule = scoringClass(model_parallel, graph, N_s=N_s, C_s=C_s, batch_size=batch_size, guide_inter=guide_inter)
        self.updateModule = updateClass(lambda_sparse=lambda_sparse, lambda_DAG=lambda_DAG)
        self.model_iters = model_iters
        self.gamma_iters = gamma_iters
        self.true_adj_matrix = torch.from_numpy(graph.adj_matrix).bool()
        self.true_node_relations = torch.from_numpy(graph.node_relations)


    def discover_graph(self, num_epochs=100, stop_early=True):
        num_stops = 0
        for epoch in track(range(num_epochs), leave=False, desc="Epoch loop"):
            self.epoch = epoch
            start_time = time.time()
            # Update Model
            self.model_fitting_step()
            self.fit_time = time.time() - start_time

            # Update gamma
            self.gamma_update_step()
            self.iter_time = time.time() - start_time

            self.print_gamma_statistics(print_gamma=False, epoch=epoch+1)

            if stop_early and (self.get_binary_adjmatrix() == self.true_adj_matrix).all():
                num_stops += 1 
                if num_stops >= 5:
                    print("Stopping early due to perfect discovery")
                    break
            else:
                num_stops = 0

        return self.gamma


    def model_fitting_step(self):
        sample_matrix = torch.sigmoid(self.gamma.detach())
        avg_loss = 0.0
        for _ in track(range(self.model_iters), leave=False, desc="Model update loop"):
            avg_loss += self.fittingModule.fit_step(sample_matrix)
        avg_loss /= self.model_iters
        # print("Average loss: %4.2f" % avg_loss)


    def gamma_update_step(self):
        for _ in track(range(self.gamma_iters), leave=False, desc="Gamma update loop"):
            self.gamma_optimizer.zero_grad()
            gammagrad, logregret, var_idx = self.scoringModule.score(self.gamma)
            self.updateModule.update(gammagrad, logregret, self.gamma, var_idx)
            self.gamma_optimizer.step()


    @torch.no_grad()
    def print_gamma_statistics(self, print_gamma=False, epoch=-1):
        m = self.get_metrics()

        if epoch > 0:
            print("--- [EPOCH %i] ---" % epoch)
        print("Gamma - Recall: %4.2f%%, Precision: %4.2f%% (TP=%i,FP=%i,FN=%i,TN=%i)" % (100.0*m["recall"], 100.0*m["precision"], m["TP"], m["FP"], m["FN"], m["TN"]))
        print("      -> FP:", ", ".join(["%s=%i" % (key,m["FP_details"][key]) for key in m["FP_details"]]))
        if print_gamma:
            rounded_gamma = torch.round(self.gamma * 100)/100
            rounded_gamma[torch.arange(self.gamma.shape[0]), torch.arange(self.gamma.shape[1])] = 0
            print(rounded_gamma.detach())

        if self.scoringModule.guide_inter:
            self.scoringModule.print_intervention_statistics()


    @torch.no_grad()
    def get_metrics(self):
        binary_gamma = self.get_binary_adjmatrix()
        TP = torch.logical_and(binary_gamma, self.true_adj_matrix).float().sum().item()
        TN = torch.logical_and(~binary_gamma, ~self.true_adj_matrix).float().sum().item()
        FP = torch.logical_and(binary_gamma, ~self.true_adj_matrix).float().sum().item()
        FN = torch.logical_and(~binary_gamma, self.true_adj_matrix).float().sum().item()
        TN = TN - self.gamma.shape[-1] # Remove diagonal as it is not being predicted

        recall = TP / max(TP + FN, 1e-5)
        precision = TP / max(TP + FP, 1e-5)

        # Get details on False Positives
        FP_elems = torch.where(torch.logical_and(binary_gamma, ~self.true_adj_matrix))
        FP_relations = self.true_node_relations[FP_elems]
        FP_dict = {
            "ancestors": (FP_relations == -1).sum().item(), # i->j => j is a child of i
            "descendants": (FP_relations == 1).sum().item(),
            "confounders": (FP_relations == 2).sum().item(),
            "independents": (FP_relations == 0).sum().item() 
        }

        metrics = {
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "recall": recall,
            "precision": precision,
            "FP_details": FP_dict
        }
        return metrics


    def get_binary_adjmatrix(self):
        return (self.gamma > 0)


    def init_gamma_params(self, num_vars, lr_gamma, betas_gamma, **kwargs):
        self.gamma = nn.Parameter(torch.zeros(num_vars, num_vars))
        self.gamma.data[torch.arange(num_vars), torch.arange(num_vars)] = -9e15
        if betas_gamma[1] > 0:
            self.gamma_optimizer = torch.optim.Adam([self.gamma], lr=lr_gamma, betas=betas_gamma)
        else:
            self.gamma_optimizer = torch.optim.SGD([self.gamma], lr=lr_gamma, momentum=betas_gamma[0])


    def to(self, device):
        self.fittingModule.model.to(device)
        self.gamma.data = self.gamma.data.to(device)


class ObservationalCategoricalData(data.Dataset):
    
    def __init__(self, graph, dataset_size):
        super().__init__()
        self.graph = graph
        self.var_names = [v.name for v in self.graph.variables]
        start_time = time.time()
        print("Creating dataset...")
        data = graph.sample(batch_size=dataset_size, as_array=True)
        self.data = torch.from_numpy(data).long()
        print("Dataset created in %4.2fs" % (time.time() - start_time))
        
    
    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, idx):
        return self.data[idx]