import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append("../")


class GraphFitting(object):


	def __init__(self, model, optimizer, data_loader):
		super().__init__()
		self.model = model 
		self.optimizer = optimizer
		self.loss_module = nn.CrossEntropyLoss()
		self.data_loader = data_loader
		self.data_iter = iter(self.data_loader)


	def _get_next_batch(self):
		try:
			batch = next(self.data_iter)
		except StopIteration:
			self.data_iter = iter(self.data_loader)
			batch = next(self.data_iter)
		return batch


	def fit_step(self, sample_matrix=None, sample_func=None):
		batch = self._get_next_batch()
		if sample_matrix is not None:
			adj_matrices = self.sample_graphs(sample_matrix, batch_size=batch.shape[0])
		else:
			adj_matrices = sample_func(batch_size=batch.shape[0])
		loss = self.train_step(batch, adj_matrices)
		return loss


	def sample_graphs(self, sample_matrix, batch_size):
		sample_matrix = sample_matrix[None].expand(batch_size, -1, -1)
		adj_matrices = torch.bernoulli(sample_matrix)
		# Mask diagonals
		adj_matrices[:,torch.arange(adj_matrices.shape[1]), torch.arange(adj_matrices.shape[1])] = 0
		return adj_matrices


	def train_step(self, inputs, adj_matrices):
		self.model.train()
		if isinstance(self.model, nn.DataParallel):
			device = "cuda:0"
		else:
			device = self.model.device
		inputs = inputs.to(device)
		adj_matrices = adj_matrices.to(device)
		# Transpose for mask because adj[i,j] means that i->j
		mask_adj_matrices = adj_matrices.transpose(1,2)
		preds = self.model(inputs, mask=mask_adj_matrices)
		loss = self.loss_module(preds.flatten(0,-2), inputs.reshape(-1))
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()