import torch
import numpy as np 
import sys
sys.path.append("../")



class GraphUpdate(object):


	def __init__(self, lambda_sparse, lambda_DAG):
		self.lambda_sparse = lambda_sparse
		self.lambda_DAG = lambda_DAG


	def update(self, gammagrad, logregret, gamma, var_idx):
		if gamma.grad is not None:
			gamma.grad.fill_(0)
		grads = self.gradient_estimator(gammagrad, logregret, var_idx)
		reg_loss = self.lambda_sparse * self.sparse_regularizer(gamma) + \
				   self.lambda_DAG * self.DAG_regularizer(gamma)
		reg_loss.backward()
		gamma.grad += grads

		# Diagonal and intervened variable set to zero
		gamma.grad[torch.arange(gamma.shape[0]), torch.arange(gamma.shape[1])] = 0.0
		gamma.grad[:,var_idx] = 0.0
		

	@torch.no_grad()
	def gradient_estimator(self, gammagrad, logregret, var_idx):
		# Bengio's paper states that L is the *probability*, not NLL as Ke's paper might suggest.
		# However, this leads to some contradictions as for the uniform distribution case.
		# For now, we use the NLL, but needs to take the negative of the gradients afterwards.
		logregret = logregret - logregret.min(dim=0)[0][None]
		logregret = (-logregret).exp() # -log X => exp(-(- log X)) = X
		# print("Probs", logregret)
		
		nomin = gammagrad * logregret.unsqueeze(dim=1) # Shape: [Batch, NumVars, NumVars]
		
		denom = logregret.unsqueeze(dim=1) # Shape: [Batch, 1, NumVars]
		nomin = nomin.sum(dim=0)
		denom = denom.sum(dim=0)
		nomin[:,var_idx] = 0.0
		denom[:,var_idx] = 1e-5

		grads = nomin / denom

		if torch.isnan(grads).any():
			print("Found NaNs")
			print("Nominator", nomin)
			print("Denominator", denom)

		# grads = - grads
		
		return grads


	def sparse_regularizer(self, gamma):
		l1 = gamma.sum()
		return l1


	def DAG_regularizer(self, gamma):
		prob_gamma = torch.sigmoid(gamma)
		cosh_matrix = torch.cosh(prob_gamma * prob_gamma.transpose(0,1))
		cosh_matrix[torch.arange(gamma.shape[0]), torch.arange(gamma.shape[1])] = 0
		return cosh_matrix.sum()