import torch
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append("../")

from DAG_generation.estimators import relax, reinforce


class GraphUpdateDAG(object):


    def __init__(self, lambda_sparse, **kwargs):
        self.lambda_sparse = lambda_sparse


    def update(self, gammagrad, logregret, gammamask, gamma, var_idx,
                     theta_z, theta_b, pl_thetas, pl_critic):
        ## GAMMA UPDATE
        if gamma.grad is not None:
            gamma.grad.fill_(0)
        grads = self.edge_gradient_estimator(gammagrad, logregret, gammamask, var_idx)
        reg_loss = self.lambda_sparse * self.sparse_regularizer(gamma)
        reg_loss.backward()
        gamma.grad += grads

        # Diagonal and intervened variable set to zero
        gamma.grad[torch.arange(gamma.shape[0]), torch.arange(gamma.shape[1])] = 0.0
        gamma.grad[:,var_idx] = 0.0

        ## THETA UPDATE
        # TODO: Test reinforce and the direction of the gradients
        # Are we actually learning what we should, or is the gradient random?
        theta_grads, _ = self.order_gradient_estimator(logregret, theta_b, pl_thetas, var_idx)
        pl_thetas.grad = theta_grads
        """
        fb = logregret.clone()
        fb[:,var_idx] = 0.0
        fb = fb.sum(dim=-1)
        print("fb", fb)
        print("b", theta_b)
        g = reinforce(fb=fb, b=theta_b, logits=pl_thetas[None].expand(theta_b.shape[0], -1))
        pl_thetas.backward(g.mean(dim=0))
        print("Thetas", pl_thetas)
        print("Gradients", g.mean(dim=0))

        fb = logregret.clone()
        fb[:,var_idx] = 0.0
        fb = fb.sum(dim=-1)
        d_log_theta = relax(fb=fb, b=theta_b, logits=pl_thetas, z=theta_z, c=pl_critic)
        (d_log_theta ** 2).mean(dim=0).sum().backward() # Loss for critic is the variance
        pl_thetas.backward(d_log_theta.mean(dim=0))
        """
        

    @torch.no_grad()
    def edge_gradient_estimator(self, gammagrad, logregret, gammamask, var_idx):
        # Bengio's paper states that L is the *probability*, not NLL as Ke's paper might suggest.
        # However, this leads to some contradictions as for the uniform distribution case.
        # For now, we use the NLL, but needs to take the negative of the gradients afterwards.
        # In the code, Ke's paper uses a softmax over mean log likelihood. Is the same as we
        # do here, just with mean instead of summing in the previous.
        logregret = logregret - logregret.min(dim=0)[0][None]
        logregret = (-logregret).exp() # -log X => exp(-(- log X)) = X
        logregret = logregret.unsqueeze(dim=1) * gammamask # Setting "masked" values to zero
        # print("Probs", logregret)
        
        nomin = gammagrad * logregret # Shape: [Batch, NumVars, NumVars]
        
        denom = logregret # Shape: [Batch, 1, NumVars]
        nomin = nomin.sum(dim=0)
        denom = denom.sum(dim=0)
        nomin[:,var_idx] = 0.0
        denom[:,var_idx] = 1e-5
        denom.masked_fill_(denom == 0.0, 1e-5)

        grads = nomin / denom

        if torch.isnan(grads).any():
            print("Found NaNs")
            print("Nominator", nomin)
            print("Denominator", denom)

        # grads = - grads
        
        return grads


    @torch.no_grad()
    def order_gradient_estimator(self, logregret, order_samples, pl_thetas, var_idx):
        # print("Thetas", pl_thetas)
        # print("Var idx", var_idx)
        # Determine gradients similar to edges. Use the same setup as in edges
        logregret = logregret - logregret.min(dim=0)[0][None]
        logregret = (-logregret).exp() # -log X => exp(-(- log X)) = X
        logregret = logregret.unsqueeze(dim=1)
        # print("Logregret", logregret)

        # Get a matrix for 1[pi_j > pi_i]
        pos = (F.one_hot(order_samples, order_samples.shape[-1]) * torch.arange(order_samples.shape[-1])[None,:,None]).sum(dim=-2)
        comp_matrix = (pos[...,None] < pos[...,None,:]).float()
        # Get a matrix for p(pi_j > pi_i). Dim=0 => theta_j, Dim=1 => theta_i
        comp_logit = F.log_softmax(torch.stack([pl_thetas[None,:].expand(pl_thetas.shape[-1],-1), # theta_i 
                                                pl_thetas[:,None].expand(-1,pl_thetas.shape[-1])  # theta_j
                                                ], dim=-1), dim=-1)[...,1]
        comp_probs = comp_logit.exp()
        # print("Order", order_samples)
        # print("Comp matrix", comp_matrix)
        # print("Comp probs", comp_probs)

        # Calculating graph weights
        theta_exp = pl_thetas.exp()
        # B - first dimension, A - second dimension, C - last dimension
        theta_A, theta_B, theta_C = theta_exp[None,:,None], theta_exp[:,None,None], theta_exp[None,None,:]
        # p(A>C|A>B)
        p_A_C_gA_B = 1 / (1 + theta_C / (theta_exp[None,:,None] + theta_exp[:,None,None]))
        factor_A_B = comp_probs[None,:,:] / p_A_C_gA_B # p(A>C) / p(A>C|A>B)
        factor_neg_A_B = (1 - comp_probs[None,:,:]) / (1 - p_A_C_gA_B) # p(C>A) / p(C>A|A>B)
        # p(A>C|B>A) 
        frac_A_B_C = theta_C * (theta_A + theta_C) / (theta_A + theta_B)
        p_A_C_gB_A = theta_A / (theta_A + theta_C + frac_A_B_C)
        factor_B_A = comp_probs[None,:,:] / p_A_C_gB_A # p(A>C) / p(A>C|B>A)
        factor_neg_B_A = (1 - comp_probs[None,:,:]) / (1 - p_A_C_gB_A) # p(C>A) / p(C>A|B>A)

        # Choose the correct A>C or C>A
        factor_comb_A_B = factor_A_B[None] * comp_matrix[:,None] + factor_neg_A_B[None] * (1 - comp_matrix[:,None])
        factor_comb_B_A = factor_B_A[None] * comp_matrix[:,None] + factor_neg_B_A[None] * (1 - comp_matrix[:,None])
        # Choose the correct A>B or B>A
        graph_weight = factor_comb_A_B * (1 - comp_matrix[...,None]) + factor_comb_B_A * comp_matrix[...,None]
        # Mask for A=B, A=C, B=C
        eye_mask = 1 - torch.eye(pl_thetas.shape[0], dtype=torch.float32, device=graph_weight.device)[None]
        eye_mask = eye_mask[:,None] * eye_mask[:,:,None] * eye_mask[:,:,:,None] 
        graph_weight = graph_weight * eye_mask + (1 - eye_mask)
        # Take product over all C
        graph_weight_red = graph_weight.prod(dim=-1) # Over C


        comp_grads = (comp_probs - comp_matrix)
        # print("Comp grads", comp_grads)
        nomin = logregret * comp_grads * graph_weight_red
        denom = logregret * graph_weight_red
        nomin, denom = nomin.sum(dim=0), denom.sum(dim=0)
        nomin[:,var_idx] = 0.0
        denom[:,var_idx] = 1e-5
        denom.masked_fill_(denom == 0.0, 1e-5)

        pairgrads = nomin / denom 

        # print("Pair grads", pairgrads)

        grads = pairgrads.sum(dim=1) - pairgrads.sum(dim=0)
        grads[var_idx] = 0.0

        # print("Grads", grads)
        # Masking gradients of intervention:
        # - Any gradients that involve the likelihood of the intervened variable are set to zero
        # - The gradient of theta_{var_idx} is set to zero. No input connection can be checked.
        #   Hence, the gradients are too biased

        debug = {
            "pairgrads": pairgrads,
            "comp_grads": comp_grads,
            "comp_probs": comp_probs,
            "comp_matrix": comp_matrix,
            "logregret": logregret,
            "nomin": nomin,
            "denom": denom,
            "graph_weight": graph_weight,
            "graph_weight_red": graph_weight_red,
            "p_A_C_gA_B": p_A_C_gA_B,
            "p_A_C_gB_A": p_A_C_gB_A,
            "factor_A_B": factor_A_B,
            "factor_neg_A_B": factor_neg_A_B,
            "factor_B_A": factor_B_A,
            "factor_neg_B_A": factor_neg_B_A,
            "factor_comb_B_A": factor_comb_B_A,
            "factor_comb_A_B": factor_comb_A_B,
            "eye_mask": eye_mask,
            "order_samples": order_samples
        }

        return grads, debug


    def sparse_regularizer(self, gamma):
        l1 = gamma.sum()
        return l1