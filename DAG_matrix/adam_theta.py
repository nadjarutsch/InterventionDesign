import torch
import torch.nn as nn
import torch.optim as optim


class OptimizerTemplate:

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    @torch.no_grad()
    def step(self):
        ## Apply update step to all parameters
        for p in self.params:
            if p.grad is None: # We skip parameters without any gradients
                continue
            self.update_param(p)

    def update_param(self, p):
        # To be implemented in optimizer-specific classes
        raise NotImplementedError


class AdamTheta(OptimizerTemplate):

    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.param_step = torch.zeros_like(self.params.data) # Remembers "t" for each parameter for bias correction
        self.param_momentum = torch.zeros_like(self.params.data)
        self.param_2nd_momentum = torch.zeros_like(self.params.data)

    def zero_grad(self):
        ## Set gradients of all parameters to zero
        if self.params.grad is not None:
            self.params.grad.detach_() # For second-order optimizers important
            self.params.grad.zero_()

    @torch.no_grad()
    def step(self, mask):
        if self.params.grad is None:
            return 

        self.param_step.add_(mask)

        new_momentum = (1 - self.beta1) * self.params.grad + self.beta1 * self.param_momentum
        new_2nd_momentum = (1 - self.beta2) * (self.params.grad)**2 + self.beta2 * self.param_2nd_momentum
        self.param_momentum = torch.where(mask == 1.0, new_momentum, self.param_momentum)
        self.param_2nd_momentum = torch.where(mask == 1.0, new_2nd_momentum, self.param_2nd_momentum)

        bias_correction_1 = 1 - self.beta1 ** self.param_step
        bias_correction_2 = 1 - self.beta2 ** self.param_step
        bias_correction_1.masked_fill_(bias_correction_1 == 0.0, 1.0)
        bias_correction_2.masked_fill_(bias_correction_2 == 0.0, 1.0)

        p_2nd_mom = self.param_2nd_momentum / bias_correction_2
        p_mom = self.param_momentum / bias_correction_1
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom
        p_update = mask * p_update

        self.params.add_(p_update)

    @torch.no_grad()
    def to(self, device):
        self.param_step = self.param_step.to(device)
        self.param_momentum = self.param_momentum.to(device)
        self.param_2nd_momentum = self.param_2nd_momentum.to(device)