import torch.nn as nn
import torch
from torchcfm.models import MLP, MLPCond
from torch.func import vmap, jacfwd, functional_call

class LinearInterpolant(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        
    def forward(self, t, x0, x1, labels=None, t_min=0, t_max=1):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        t_max = t_max.reshape(-1, *([1] * (x0.dim() - 1)))
        t_min = t_min.reshape(-1, *([1] * (x0.dim() - 1)))
        return (t - t_min) / (t_max - t_min) * x1 + (t_max - t) / (t_max - t_min) * x0
        
    def gradient(self, t, x0, x1, labels=None, t_min=0, t_max=1):
        return (x1 - x0) / (t_max - t_min)
    
    def get_interp_and_phi(self, t, x0, x1, labels, t_min=0, t_max=1):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        t_adj = (t - t_min) / (t_max - t_min)
        return t_adj * x1 + (1-t_adj) * x0, None

class CondInterpolant(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        def fmodel(params, t, x0, x1, labels, t_min=0, t_max=1):
            return functional_call(self, params, (t, x0, x1, labels, t_min, t_max))
        self.model_grad = vmap(jacfwd(fmodel, argnums=(1,)), in_dims=(None,0,0,0,0,0,0))

    def forward(self, t, x0, x1, labels, t_min=0, t_max=1):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        t_adj = (t - t_min) / (t_max - t_min)
        return t_adj * x1 + (1-t_adj) * x0 + t_adj * (1-t_adj) * self.model(t, x0, x1, labels)

    def get_interp_and_phi(self, t, x0, x1, labels, t_min=0, t_max=1):
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        t_adj = (t - t_min) / (t_max - t_min)
        phi = self.model(t, x0, x1, labels)
        return t_adj * x1 + (1-t_adj) * x0 + t_adj * (1-t_adj) * phi, phi

    def gradient(self, t, x0, x1, labels, t_min=None, t_max=None):
        # convert [1] shape to [512, 1]
        if t_min is None:
            t_min = torch.zeros_like(t)
        else:
            t_min = t_min.expand(t.shape[0], 1)  # or use t_min.repeat(t.shape[0], 1)
        if t_max is None:
            t_max = torch.ones_like(t)
        else:
            t_max = t_max.expand(t.shape[0], 1)  # or use t_max.repeat(t.shape[0], 1)
        return self.model_grad(dict(self.named_parameters()), t, x0, x1, labels, t_min, t_max)[0].squeeze(1).squeeze(-1).squeeze(-1)
