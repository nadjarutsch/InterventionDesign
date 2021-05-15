import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
import sys
sys.path.append("../")
from causal_discovery.model_utils import get_device


class ProbDist(object):

	def __init__(self):
		pass

	def sample(self, inputs, batch_size=1):
		raise NotImplementedError

	def prob(self, inputs, output):
		raise NotImplementedError


####################
## DISCRETE PROBS ##
####################

class DiscreteProbDist(ProbDist):

	def __init__(self, val_range):
		super().__init__()
		self.val_range = val_range


class ConstantDist(DiscreteProbDist):

	def __init__(self, constant, val_range=None, **kwargs):
		super().__init__(val_range=val_range)
		self.constant = constant 

	def sample(self, inputs, batch_size=1):
		return np.repeat(self.constant, batch_size) 

	def prob(self, inputs, output):
		return 1 if output == self.constant else 0

	def get_state_dict(self):
		state_dict = vars(self)
		return state_dict

	@staticmethod
	def load_from_state_dict(state_dict):
		obj = ConstantDist(state_dict["constant"], state_dict["val_range"])
		return obj


class CategoricalDist(DiscreteProbDist):

	def __init__(self, num_categs, prob_func, **kwargs):
		super().__init__(val_range=(0,num_categs))
		self.num_categs = num_categs
		self.prob_func = prob_func

	def sample(self, inputs, batch_size=1):
		p = self.prob_func(inputs, batch_size)
		if len(p.shape) == 1:
			p = np.repeat(p[None], batch_size, axis=0)
		v = multinomial_batch(p)
		return v

	def prob(self, inputs, output):
		p = self.prob_func(inputs, batch_size=1)
		while len(p.shape) > 2:
			p = p[0]
		if len(p.shape) == 2:
			return p[np.arange(output.shape[0]), output]
		else:
			return p[...,output]

	def get_state_dict(self):
		state_dict = {"num_categs": self.num_categs,
					  "prob_func": self.prob_func.get_state_dict()}
		state_dict["prob_func"]["class_name"] = str(self.prob_func.__class__.__name__)
		return state_dict 

	@staticmethod
	def load_from_state_dict(state_dict):
		prob_func_class = None
		if state_dict["prob_func"]["class_name"] == "LeafCategDist":
			prob_func_class = LeafCategDist
		elif state_dict["prob_func"]["class_name"] == "CategProduct":
			prob_func_class = CategProduct
		elif state_dict["prob_func"]["class_name"] == "IndependentCategProduct":
			prob_func_class = IndependentCategProduct
		elif state_dict["prob_func"]["class_name"] == "NNCateg":
			prob_func_class = NNCateg
		prob_func = prob_func_class.load_from_state_dict(state_dict["prob_func"])
		obj = CategoricalDist(state_dict["num_categs"], prob_func)
		return obj


class LeafCategDist:

	def __init__(self, num_categs):
		self.probs = _random_categ(size=(num_categs,))
		self.num_categs = num_categs

	def __call__(self, inputs, batch_size):
		return self.probs

	def get_state_dict(self):
		state_dict = copy(vars(self))
		return state_dict

	@staticmethod
	def load_from_state_dict(state_dict):
		obj = LeafCategDist(state_dict["num_categs"])
		obj.probs = state_dict["probs"]
		return obj


class CategProduct:

	def __init__(self, input_names, input_num_categs=None, num_categs=None, val_grid=None):
		if val_grid is None:
			assert input_num_categs is not None and num_categs is not None
			val_grid = _random_categ(size=tuple(input_num_categs) + (num_categs,))
		else:
			num_categs = val_grid.shape[-1]
			input_num_categs = val_grid.shape[:-1]
		self.val_grid = val_grid
		self.input_names = input_names
		self.input_num_categs = input_num_categs
		self.num_categs = num_categs
		
	def __call__(self, inputs, batch_size):
		idx = tuple([inputs[name] for name in self.input_names])
		v = self.val_grid[idx]
		return v

	def get_state_dict(self):
		state_dict = copy(vars(self))
		return state_dict 

	@staticmethod
	def load_from_state_dict(state_dict):
		obj = CategProduct(state_dict["input_names"], 
                           state_dict["input_num_categs"], 
                           state_dict["num_categs"])
		obj.val_grid = state_dict["val_grid"]
		return obj


class IndependentCategProduct:

	def __init__(self, input_names, input_num_categs, num_categs,   
					   scale_prior=0.2, scale_val=1.0):
		num_inputs = len(input_names)
		val_grids = [_random_categ(size=(c, num_categs), scale=scale_val) for c in input_num_categs]
		prior = _random_categ((num_inputs,), scale=scale_prior)
		self.val_grids = val_grids
		self.prior = prior
		self.num_categs = num_categs
		self.input_names = input_names
		self.input_num_categs = input_num_categs

	def __call__(self, inputs, batch_size):
		probs = np.zeros((batch_size, self.num_categs))
		for idx, name in enumerate(self.input_names):
			probs += self.prior[idx] * self.val_grids[idx][inputs[name]]
		return probs

	def get_state_dict(self):
		state_dict = copy(vars(self))
		return state_dict

	@staticmethod
	def load_from_state_dict(state_dict):
		obj = IndependentCategProduct(state_dict["input_names"], 
			                          state_dict["input_num_categs"], 
			                          state_dict["num_categs"])
		obj.prior = state_dict["prior"]
		obj.val_grids = state_dict["val_grids"]
		return obj


class NNCateg:

	def __init__(self, input_names, input_num_categs, num_categs):
		num_hidden = 48 # max(4*num_categs, 4*len(input_names))
		embed_dim = 4 # min(4, num_hidden//len(input_num_categs))
		self.embed_module = nn.Embedding(sum(input_num_categs), embed_dim)
		# nn.init.orthogonal_(self.embed_module.weight, gain=2.5)
		self.net = nn.Sequential(nn.Linear(embed_dim*len(input_num_categs), num_hidden),
							     nn.LeakyReLU(0.1),
							     nn.Linear(num_hidden, num_categs, bias=False),
							     nn.Softmax(dim=-1))
		for name, p in self.net.named_parameters():
			if name.endswith(".bias"):
				if name.startswith("2."):
					p.data.fill_(0.0)
				else:
					p.data.uniform_(-0.5,0.5) # Changed from -1.1,1.1 to -0.5,0.5
				# p.data.uniform_(-1.1,1.1)
			else:
				nn.init.orthogonal_(p, gain=2.5)
		self.num_categs = num_categs
		self.input_names = input_names
		self.input_num_categs = input_num_categs
		self.device = get_device()
		self.embed_module.to(self.device)
		self.net.to(self.device)

	@torch.no_grad()
	def __call__(self, inputs, batch_size):
		inp_tensor = None
		for i, n, categs in zip(range(len(self.input_names)), self.input_names, self.input_num_categs):
			v = torch.from_numpy(inputs[n]).long()+sum(self.input_num_categs[:i])
			v = v.unsqueeze(dim=-1)
			inp_tensor = v if inp_tensor is None else torch.cat([inp_tensor, v], dim=-1)
		inp_tensor = inp_tensor.to(self.device)
		inp_tensor = self.embed_module(inp_tensor).flatten(-2,-1)
		probs = self.net(inp_tensor).cpu().numpy()
		return probs

	def get_state_dict(self):
		state_dict = copy(vars(self))
		state_dict["embed_module"] = self.embed_module.state_dict()
		state_dict["net"] = self.net.state_dict()
		return state_dict

	@staticmethod
	def load_from_state_dict(state_dict):
		obj = NNCateg(state_dict["input_names"], 
                      state_dict["input_num_categs"], 
                      state_dict["num_categs"])
		obj.embed_module.load_state_dict(state_dict["embed_module"])
		obj.net.load_state_dict(state_dict["net"])
		return obj



def multinomial_batch(p):
	u = np.random.uniform(size=p.shape[:-1]+(1,))
	p_cumsum = np.cumsum(p, axis=-1)
	diff = (p_cumsum - u)
	diff[diff < 0] = 2 # Set negatives to any number larger than 1
	samples = np.argmin(diff, axis=-1)
	return samples



######################
## CONTINUOUS PROBS ##
######################

class ContinuousProbDist(ProbDist):

	def __init__(self):
		super().__init__()


class GaussianDist(ContinuousProbDist):

	def __init__(self, mu_func, sigma_func, max_val=10, **kwargs):
		super().__init__()
		self.mu_func = mu_func
		self.sigma_func = sigma_func
		self.max_val = max_val


	def _get_params(self, inputs):
		mu = self.mu_func(inputs)
		sigma = self.sigma_func(inputs)
		mu = np.clip(mu, a_min=-self.max_val, a_max=self.max_val)
		return mu, sigma


	def sample(self, inputs, batch_size=1):
		mu, sigma = self._get_params(inputs)
		if not isinstance(mu, np.ndarray) or mu.shape[0] == 1:
			mu = np.repeat(mu, batch_size, axis=0)
			sigma = np.repeat(sigma, batch_size, axis=0)
		eps = np.random.normal(mu, sigma)
		return eps


	def prob(self, inputs, output, *args, **kwargs):
		mu, sigma = self._get_params(inputs)
		if len(mu.shape) == 1 and len(output.shape) == 2:
			mu = np.repeat(mu[None], output.shape[0], axis=0)
			sigma = np.repeat(sigma[None], output.shape[0], axis=0)
		p = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(output-mu)**2/(2*sigma**2))
		return p 


#####################
## DIST GENERATORS ##
#####################

def _random_categ(size, scale=1.0, axis=-1):
	val_grid = np.random.normal(scale=scale, size=size)
	val_grid = np.exp(val_grid)
	val_grid = val_grid / val_grid.sum(axis=axis, keepdims=True)
	return val_grid


def get_random_categorical(input_names, input_num_categs, num_categs, inputs_independent=True, use_nn=False, **kwargs):
	num_inputs = len(input_names)

	if num_inputs == 0:
		prob_func = LeafCategDist(num_categs)
	elif use_nn:
		prob_func = NNCateg(input_names, input_num_categs, num_categs)
	elif inputs_independent:
		prob_func = IndependentCategProduct(input_names, input_num_categs, num_categs)
	else:
		prob_func = CategProduct(input_names, input_num_categs, num_categs)

	return CategoricalDist(num_categs, prob_func, **kwargs)


def get_random_gaussian(input_names, num_coeff=4, **kwargs):
	if len(input_names) > 0:
		mu_funcs = {}
		std_funcs = {}

		for name in input_names:
			# Mean
			mu_coeff = [np.random.normal(loc=0.0, scale=1.0/(i+1)**(1.8)) for i in range(num_coeff)]
			mu_funcs[name] = lambda x : sum([c * x**i for i, c in enumerate(mu_coeff)]) * np.exp(-(x/4)**2)
			# STD
			std_coeff = [np.random.normal(loc=0.0, scale=1.0/(i+1)**(1.8)) for i in range(num_coeff)]
			std_funcs[name] = lambda x : np.exp(np.tanh(sum([c * x**i for i, c in enumerate(std_coeff)])))

		mu_func = lambda inputs : sum([mu_funcs[name](inputs[name]) for name in inputs])/len(inputs)
		def sigma_func(inputs):
			sigmas = [std_funcs[name](inputs[name]) for name in inputs]
			sigma = 1.0
			for s in sigmas:
				sigma *= s
			if len(sigmas) > 0:
				sigma = sigma ** (1.0 / len(sigmas))
			return sigma
	else:
		mu, sigma = np.random.normal(loc=0.0, scale=2.0), np.exp(np.random.normal(loc=0.0, scale=0.5))
		mu_func = lambda inputs : mu 
		sigma_func = lambda inputs: sigma

	return GaussianDist(mu_func, sigma_func, **kwargs)