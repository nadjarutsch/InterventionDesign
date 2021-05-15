import torch
import torch.nn as nn
import torch.utils.data as data 
import torch.nn.functional as F
import numpy as np 
import random
import time
import sys
sys.path.append("../")

from causal_graphs.variable_distributions import _random_categ


class GraphScoring(object):


	def __init__(self, model, graph, N_s, C_s, batch_size, guide_inter=False):
		self.model = model
		self.graph = graph
		self.N_s = N_s
		self.C_s = C_s
		self.batch_size = batch_size
		self.guide_inter = guide_inter
		self.stats = np.zeros((self.graph.num_vars,), dtype=np.int32)
		self.inter_vars = []


	@torch.no_grad()
	def score(self, gamma):
		intervention_dict, var_idx = self.sample_intervention(self.graph, 
															  dataset_size=self.N_s*self.batch_size,
															  gamma=gamma)
		int_sample = self.graph.sample(interventions=intervention_dict, 
			                           batch_size=self.N_s*self.batch_size, 
			                           as_array=True)
		int_sample = torch.from_numpy(int_sample)

		gammagrad = []
		logregret = []
		sample_matrix = torch.sigmoid(gamma[None]).detach()
		if torch.isnan(sample_matrix).any():
			print("Sample matrix", sample_matrix)
			print("Gamma", gamma)
			print("Grads", gamma.grad)
		for n_idx in range(self.N_s):
			batch = torch.LongTensor(int_sample[n_idx*self.batch_size:(n_idx+1)*self.batch_size])
			for c_idx in range(self.C_s):
				adj_matrix = torch.bernoulli(sample_matrix).expand(self.batch_size, -1, -1)
				nll = self.evaluate_likelihoods(batch, adj_matrix, var_idx)
				gammagrad.append(sample_matrix - adj_matrix)
				logregret.append(nll)

				gammagrad[-1] = gammagrad[-1][0:1]
				logregret[-1] = logregret[-1].sum(dim=0, keepdim=True)
				# print("Sampled adj matrix:\n", adj_matrix[0])
				# print("NLL", nll.mean(dim=0))
		gammagrad = torch.cat(gammagrad, dim=0)
		logregret = torch.cat(logregret, dim=0)

		return gammagrad, logregret, var_idx


	def sample_intervention(self, graph, dataset_size, gamma, var_idx=-1):
		if var_idx >= 0:
			pass
		elif not self.guide_inter:
			var_idx = self.sample_next_var_idx()
		else:
			true_adj_matrix = torch.from_numpy(self.graph.adj_matrix).float()
			dist = (true_adj_matrix - torch.sigmoid(gamma)).abs()
			var_idx = torch.argmax(dist.max(dim=1).values, dim=0).item()
		var = graph.variables[var_idx]
		self.stats[var_idx] += 1 # Recording on which variable we intervened
		# Soft intervention => replace p(X_n) by random categorical
		int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)
		value = np.random.multinomial(n=1, pvals=int_dist, size=(dataset_size,))
		value = np.argmax(value, axis=-1) # One-hot to index
		# value[:] = 0
		intervention_dict = {var.name: value}
		return intervention_dict, var_idx


	def sample_next_var_idx(self):
		if len(self.inter_vars) == 0:
			self.inter_vars = [i for i in range(len(self.graph.variables))]
			random.shuffle(self.inter_vars)
		var_idx = self.inter_vars.pop()
		return var_idx



	@torch.no_grad()
	def evaluate_likelihoods(self, int_sample, adj_matrix, var_idx):
		self.model.eval()
		device = self.get_device()
		int_sample = int_sample.to(device)
		adj_matrix = adj_matrix.to(device)
		# Transpose for mask because adj[i,j] means that i->j
		mask_adj_matrix = adj_matrix.transpose(1,2)
		preds = self.model(int_sample, mask=mask_adj_matrix)
		
		preds = preds.flatten(0, 1)
		labels = int_sample.clone()
		labels[:,var_idx] = -1
		labels = labels.reshape(-1)
		ll = F.cross_entropy(preds, labels, reduction='none', ignore_index=-1)
		ll = ll.reshape(*int_sample.shape)
		self.model.train()
		return ll


	def get_device(self):
		if isinstance(self.model, nn.DataParallel):
			device = "cuda:0"
		else:
			device = self.model.device
		return device


	def print_intervention_statistics(self):
		print("Interventions: " + ", ".join(["%s: %i" % (self.graph.variables[i].name, self.stats[i]) for i in range(self.graph.num_vars)]))