import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import numpy as np
import time


class MultivarLinear(nn.Module):

	def __init__(self, c_in, c_out, extra_dims):
		super().__init__()
		self.c_in = c_in
		self.c_out = c_out
		self.extra_dims = extra_dims

		self.weight = nn.Parameter(torch.zeros(*extra_dims, c_out, c_in))
		self.bias = nn.Parameter(torch.zeros(*extra_dims, c_out))

		nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')


	def forward(self, x):
		# Shape preparation
		batch_size, c_in = x.shape[0], x.shape[-1]
		x_extra_dims = x.shape[1:-1]
		if len(x_extra_dims) > 0:
			for i in range(len(x_extra_dims)):
				assert x_extra_dims[-(i+1)] == self.extra_dims[-(i+1)], \
					   "Shape mismatch: X=%s, Layer=%s" % (str(x.shape), str(self.extra_dims))
		for _ in range(len(self.extra_dims)-len(x_extra_dims)):
			x = x.unsqueeze(dim=1)

		# Unsqueeze
		x = x.unsqueeze(dim=-1)
		weight = self.weight.unsqueeze(dim=0)
		bias = self.bias.unsqueeze(dim=0)

		# Linear layer
		out = torch.matmul(weight, x).squeeze(dim=-1)
		out = out + bias
		return out


	def extra_repr(self):
		return 'c_in={}, c_out={}, extra_dims={}'.format(
		    self.c_in, self.c_out, str(self.extra_dims)
		)


class InputMask(nn.Module):

	def __init__(self, input_mask):
		super().__init__()
		if isinstance(input_mask, torch.Tensor):
			self.register_buffer('input_mask', input_mask.float(), persistent=False)
		else:
			self.input_mask = input_mask


	def forward(self, x, mask=None, mask_val=0):
		if mask is None:
			assert self.input_mask is not None, "No mask was given in InputMask module."
			mask = self.input_mask

		if len(mask.shape) > len(x.shape):
			x = x.reshape(x.shape[:1] + (1,)*(len(mask.shape)-len(x.shape)) + x.shape[1:])
		if len(x.shape) > len(mask.shape):
			mask = mask.reshape((1,)*(len(x.shape)-len(mask.shape)) + mask.shape)
		mask = mask.to(x.dtype)

		x = x * mask + (1 - mask) * mask_val
		return x


class EmbedLayer(nn.Module):

	def __init__(self, num_vars, num_categs, hidden_dim, input_mask, share_embeds=False, stack_embeds=False, sparse_embeds=False):
		super().__init__()
		self.num_vars = num_vars
		self.hidden_dim = hidden_dim
		self.input_mask = input_mask
		self.share_embeds = share_embeds
		self.stack_embeds = stack_embeds
		self.sparse_embeds = sparse_embeds
		self.num_categs = num_categs+1 if not sparse_embeds else num_categs
		self.num_embeds = self.num_vars*self.num_vars*self.num_categs if not self.share_embeds else self.num_vars*self.num_categs
		if self.num_embeds > 1e6:
			self.num_embeds = self.num_embeds // 10
			self.shortend = True
		else:
			self.shortend = False
		self.embedding = nn.Embedding(num_embeddings=self.num_embeds,
									  embedding_dim=hidden_dim)
		self.embedding.weight.data.mul_(2./math.sqrt(self.num_vars))
		self.bias = nn.Parameter(torch.zeros(num_vars, self.output_dim))

		pos_trans = torch.arange(self.num_vars if self.share_embeds else self.num_vars**2, dtype=torch.long) * self.num_categs
		self.register_buffer("pos_trans", pos_trans, persistent=False)
		if not sparse_embeds:
			if self.shortend:
				self.embedding.weight.data[pos_trans[:self.num_embeds//self.num_categs]+self.num_categs-1,:] = 0.
			else:
				self.embedding.weight.data[pos_trans+self.num_categs-1,:] = 0.


	def forward(self, x, mask):
		num_chunks = int(math.ceil(np.prod(mask.shape) / 256e5))
		if self.training or num_chunks == 1:
			return self.embed_tensor(x, mask)
		else:
			x = x.chunk(num_chunks, dim=0)
			mask = mask.chunk(num_chunks, dim=0)
			x_out = []
			for x_l, mask_l in zip(x, mask):
				out_l = self.embed_tensor(x_l, mask_l)
				x_out.append(out_l)
			x_out = torch.cat(x_out, dim=0)
			return x_out

		

	def embed_tensor(self, x, mask):
		assert x.shape[-1] == self.num_vars
		if self.share_embeds:
			# Number of variables
			pos_trans = self.pos_trans.view((1,)*(len(x.shape)-1) + (self.num_vars,))
			x = x + pos_trans

			embeds = self.embedding(x.long())
			if len(mask.shape) > len(x.shape):
				embeds = embeds.reshape(embeds.shape[:1] + (1,)*(len(mask.shape)-len(x.shape)) + embeds.shape[1:])
			default = self.embedding(pos_trans + self.num_categs - 1)[None]
			
			x = torch.where(mask[...,None] == 1.0, embeds, default)
		else:
			if len(x.shape) == 2: # Add variable dimension
				x = x.unsqueeze(dim=1).expand(-1,self.num_vars,-1)
			else:
				assert x.shape[-2] == self.num_vars

			# Number of variables
			pos_trans = self.pos_trans.view((1,)*(len(x.shape)-2) + (self.num_vars, self.num_vars))
			x = x + pos_trans

			if self.sparse_embeds:
				# print("Mask", mask)
				flattened_mask = mask.flatten(0,1).long()
				num_neighbours = flattened_mask.sum(dim=-1)
				max_neighbours = num_neighbours.max()
				# print("Num neighbours", num_neighbours)
				# print("Max neighbours", max_neighbours)
				# print("Avg neighbours", num_neighbours.float().mean())
				# print("Costs", comp_cost, "Neighbours", sort_neighbours)
				# print("Minimum cost", min_cost, "Max cost", num_neighbours.shape[0]*max_neighbours)
				# print("X", x)
				x_sparse = torch.masked_select(x, mask == 1.0)
				if self.shortend:
					x_sparse = x_sparse % self.num_embeds
				# print("X sparse (1)", x_sparse.shape, x_sparse)
				x_sparse = self.embedding(x_sparse)
				x_sparse = torch.cat([x_sparse.new_zeros(x_sparse.shape[:-2]+(1,)+x_sparse.shape[-1:]), x_sparse], dim=-2)
				# print("X sparse (2)", x_sparse.shape)
				idxs = flattened_mask.cumsum(dim=-1)
				# print("Idxs (0)", idxs)
				idxs[1:] += num_neighbours[:-1].cumsum(dim=-1)[...,None]
				# print("Idxs (1)", idxs)
				idxs = (idxs * flattened_mask).sort(dim=-1, descending=True)[0]
				# print("Idxs (2)", idxs)
				if True:
					sort_neighbours, sort_indices = num_neighbours.sort(dim=0)
					_, resort_indices = sort_indices.sort(dim=0)
					pos = 1+torch.arange(num_neighbours.shape[0], device=num_neighbours.device, dtype=torch.long)
					comp_cost = sort_neighbours * pos + max_neighbours * (num_neighbours.shape[0] - pos)
					min_cost, argmin_cost = comp_cost.min(dim=0)
					mid_neighbours = sort_neighbours[argmin_cost]
					
					idxs = idxs[sort_indices]
					idxs = idxs[:,:max_neighbours]
					# print("Idxs (3)", idxs)
					if mid_neighbours > 0:
						x_new_1 = x_sparse.index_select(index=idxs[:argmin_cost+1,:mid_neighbours].reshape(-1), dim=0)
						x_1 = x_new_1.reshape(-1, mid_neighbours, x_sparse.shape[-1]).sum(dim=-2)
					else:
						x_1 = x_sparse.new_zeros(argmin_cost+1, x_sparse.shape[-1])
					x_new_2 = x_sparse.index_select(index=idxs[argmin_cost+1:,:max_neighbours].reshape(-1), dim=0)
					x_2 = x_new_2.reshape(-1, max_neighbours, x_sparse.shape[-1]).sum(dim=-2)
					x = torch.cat([x_1, x_2], dim=0)[resort_indices]
				else:
					x = x_sparse.index_select(index=idxs[:,:max_neighbours].reshape(-1), dim=0)
					x = x.reshape(-1, max_neighbours, x_sparse.shape[-1]).sum(dim=-2)
				x = x.reshape(mask.shape[0], mask.shape[1], x.shape[-1])
			else:
				x = self.input_mask(x, mask=mask, mask_val=pos_trans + self.num_categs - 1).long()
				if self.shortend:
					x = x % self.num_embeds
				x = self.embedding(x)

		if len(x.shape) > 3:
			if self.stack_embeds:
				x = x.flatten(-2, -1)
			else:
				x = x.sum(dim=-2)

		bias = self.bias.view((1,)*(len(x.shape)-2) + self.bias.shape)
		x = x + bias
		return x


	@property
	def output_dim(self):
		return self.hidden_dim if not self.stack_embeds else (self.hidden_dim * self.num_vars)


class MultivarMLP(nn.Module):

	def __init__(self, input_dims, hidden_dims, output_dims, extra_dims, actfn, pre_layers=None):
		super().__init__()
		self.extra_dims = extra_dims

		layers = []
		if pre_layers is not None:
			if not isinstance(pre_layers, list):
				layers += [pre_layers]
			else:
				layers += pre_layers
		hidden_dims = [input_dims] + hidden_dims
		for i in range(len(hidden_dims)-1):
			if not isinstance(layers[-1], EmbedLayer) or layers[-1].stack_embeds:
				layers += [MultivarLinear(c_in=hidden_dims[i],
										  c_out=hidden_dims[i+1],
										  extra_dims=extra_dims)]
			layers += [actfn()]
		layers += [MultivarLinear(c_in=hidden_dims[-1],
								  c_out=output_dims,
								  extra_dims=extra_dims)]
		self.layers = nn.ModuleList(layers)


	def forward(self, x, mask=None):
		for l in self.layers:
			if isinstance(l, (EmbedLayer, InputMask)):
				x = l(x, mask=mask)
			else:
				x = l(x)
		return x


	@property
	def device(self):
		return next(iter(self.parameters())).device


def get_activation_function(actfn):
	if actfn is None or actfn == 'leakyrelu':
		actfn = lambda : nn.LeakyReLU(0.1, inplace=True)
	elif actfn == 'gelu':
		actfn = lambda : nn.GELU()
	elif actfn == 'relu':
		actfn = lambda : nn.ReLU()
	elif actfn == 'swish' or actfn == 'silu':
		actfn = lambda : nn.SiLU()
	return actfn
	

def create_model(num_vars, num_categs, hidden_dims, share_embeds=False, actfn=None, sparse_embeds=False):
	num_outputs = max(1, num_categs)
	num_inputs = num_vars
	actfn = get_activation_function(actfn)

	mask = InputMask(None)
	if num_categs > 0:
		pre_layers = EmbedLayer(num_vars=num_vars,
								num_categs=num_categs,
								hidden_dim=hidden_dims[0],
								input_mask=mask,
								share_embeds=share_embeds,
								sparse_embeds=sparse_embeds)
		num_inputs = pre_layers.output_dim
		pre_layers = [pre_layers, actfn()]
	else:
		pre_layers = mask

	mlps = MultivarMLP(input_dims=num_inputs, 
	                   hidden_dims=hidden_dims, 
	                   output_dims=num_outputs, 
	                   extra_dims=[num_vars],
	                   actfn=actfn,
	                   pre_layers=pre_layers)
	return mlps