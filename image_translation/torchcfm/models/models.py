import torch


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        pe = torch.zeros(x.size(0), x.size(1), self.d_model).to(x.device)
        position = x.unsqueeze(-1) * 100
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(torch.log(torch.tensor(10000.0)) / self.d_model)).to(x.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe.squeeze(1)

class MLPCond(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, hidden_layers=2, num_conditionals=1, embedding_dim=32, use_sinusoidal_positional_embedding=False, d_model=64):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        
        if num_conditionals > 1:
            self.embedding_layer = torch.nn.Embedding(num_embeddings=num_conditionals, embedding_dim=embedding_dim)
        self.num_conditionals = num_conditionals
        
        self.use_sinusoidal = use_sinusoidal_positional_embedding
        if self.use_sinusoidal:
            self.pos_embedding = SinusoidalPositionalEmbedding(d_model=d_model)
            
        input_dim = dim + (d_model if self.use_sinusoidal else 1 if time_varying else 0) + (embedding_dim if num_conditionals > 1 else 0)
        modules = [torch.nn.Linear(input_dim, w),
                torch.nn.SELU()]
        for i in range(hidden_layers):
            modules.append(torch.nn.Linear(w, w))
            modules.append(torch.nn.SELU())
        modules.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*modules)
       
    def forward(self, t, x, labels=None):
        if self.use_sinusoidal:
            t_emb = self.pos_embedding(t)
            if self.num_conditionals > 1:
                x = torch.cat([x, t_emb, self.embedding_layer(labels)], dim=-1)
            else:
                x = torch.cat([x, t_emb], dim=-1)
        else:
            if self.num_conditionals > 1:
                x = torch.cat([x, t, self.embedding_layer(labels)], dim=-1)
            else:
                x = torch.cat([x, t], dim=-1)
        return self.net(x)

class MLPInterpolantCond(MLPCond):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, t, x0, x1, labels=None):
        return super().forward(t, torch.cat([x0, x1], dim=-1), labels)

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, hidden_layers=2, use_sinusoidal_positional_embedding=False, d_model=64):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
            
        self.use_sinusoidal = use_sinusoidal_positional_embedding
        if self.use_sinusoidal:
            self.pos_embedding = SinusoidalPositionalEmbedding(d_model=d_model)
            input_dim = dim + (d_model if self.use_sinusoidal else 1 if time_varying else 0)
        else:
            input_dim = dim + (1 if time_varying else 0)

        modules = [torch.nn.Linear(input_dim, w),
                torch.nn.SELU()]
        for i in range(hidden_layers):
            modules.append(torch.nn.Linear(w, w))
            modules.append(torch.nn.SELU())
        modules.append(torch.nn.Linear(w, out_dim))
        self.net = torch.nn.Sequential(*modules)
       
    def forward(self, t, x):
        if self.use_sinusoidal:
            t_emb = self.pos_embedding(t)
            x = torch.cat([x, t_emb], dim=-1)
        else:
            x = torch.cat([x, t], dim=-1) if self.time_varying else x
        return self.net(x)


class MLP_flow(torch.nn.Module):
    def __init__(self, dim, out_dim=None, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        w = dim + (1 if time_varying else 0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


    
class MLP2(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.ReLU(),
            torch.nn.Linear(w, w),
            torch.nn.ReLU(),
            torch.nn.Linear(w, w),
            torch.nn.ReLU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]
