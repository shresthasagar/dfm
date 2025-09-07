import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdyn
from torchdyn.datasets import generate_moons
from torchdyn.core import NeuralODE

# Implement some helper functions


def two_gaussians(n, dim, scale=1, var=1, cond=False):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )

    centers = [
        (0, 1),
        (0, -1),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(2), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    if not cond:
        return data
    else:
        return data, multi


def two_gaussians_cross(n, dim, scale=1, var=1, cond=False, first=True):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )

    if first:
        centers = [
            [1,  1],
            [1, -1],
        ]
    else:
        centers = [
            [-1, -1],
            [-1,  1],
        ]
    
    if dim>2:
        centers = [center + [0]*(dim-2) for center in centers]

    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(2), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    if not cond:
        return data
    else:
        return data.float(), multi



def eight_normal_sample(n, dim, scale=1, var=1, cond=False):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    # centers = [
    #     (1, 0),
    #     (-1, 0),
    #     (0, 1),
    #     (0, -1),
    #     (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
    #     (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    #     (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
    #     (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    # ]

    centers = [
        (1, 0),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (0, 1),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1, 0),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (0, -1),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    if not cond:
        return data
    else:
        return data, multi


def sample_moons(n, cond=False):
    x0, y0 = generate_moons(n, noise=0.2)
    if not cond:
        return x0 * 3 - 1
    else:
        return x0 * 3 - 1, y0

def sample_8gaussians(n, cond=False):
    out = eight_normal_sample(n, 2, scale=5, var=0.1, cond=cond)
    if cond:
        return out[0].float(), out[1]
    else:
        return out.float()

def sample_2gaussians(n, cond=False):
    out = two_gaussians(n, 2, scale=5, var=0.1, cond=cond)
    if cond:
        return out[0].float(), out[1]
    else:
        return out.float()

# class torch_wrapper(torch.nn.Module):
#     """Wraps model to torchdyn compatible format."""

#     def __init__(self, model, labels=None):
#         super().__init__()
#         self.model = model
#         self.labels = labels

#     def forward(self, t, x, *args, **kwargs):
#         if self.labels is not None:
#             return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1), self.labels)
#         else:
#             return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


# class torch_wrapper_unet_unified(torch.nn.Module):
#     """Wraps model to torchdyn compatible format."""

#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, t, x, *args, **kwargs):
#         return self.model(x, t.repeat(x.shape[0])[:, None])


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, labels=None):
        super().__init__()
        self.model = model
        self.labels = labels

    def forward(self, t, x, *args, **kwargs):
        if self.labels is not None:
            return self.model(t.repeat(x.shape[0])[:, None], x, self.labels)
        else:
            return self.model(t.repeat(x.shape[0])[:, None], x)


class torch_wrapper_mlp(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, labels=None):
        super().__init__()
        self.model = model
        self.labels = labels

    def forward(self, t, x, *args, **kwargs):
        if self.labels is not None:
            return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1), self.labels)
        else:
            return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))

          
def plot_trajectories(traj, ticks_off=True, x1=None):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="blue")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive", label='_nolegend_')
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=20, alpha=0.8, c="blue", marker='x')
    
    if x1 is not None:
        plt.scatter(x1[:n, 0], x1[:n, 1], s=20, alpha=0.3, c="black")
        plt.legend([r"$\boldsymbol{x}$", r"$\widehat{\boldsymbol{y}}$", r"$\boldsymbol{y}$"], fontsize=12)
    else:
        plt.legend([r"$\boldsymbol{x}$", r"$\widehat{\boldsymbol{y}}$"], fontsize=12)
    # Make axes equal to prevent squeezing
    plt.axis('equal')
    
    if ticks_off:
        plt.xticks([])
        plt.yticks([])
    image = plt.gcf()
    plt.show()
    return image


def plot_trajectories_pred_target(traj, ticks_off=True):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.5, c="black")
    plt.scatter(traj[:-2, :n, 0], traj[:-2, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-2, :n, 0], traj[-2, :n, 1], s=4, alpha=0.5, c="blue")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=0.5, c="red")

    plt.legend(["source", "Flow", "prediction", "target"])
    if ticks_off:
        plt.xticks([])
        plt.yticks([])
    image = plt.gcf()
    plt.show()
    return image

def plot_multiple_trajectories(trajs, ticks=True):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    # for traj in trajs:
    plt.scatter(trajs[0][0, :n, 0], trajs[0][0, :n, 1], s=15, alpha=0.8, c="blue")
    plt.scatter(trajs[0][:, :n, 0], trajs[0][:, :n, 1], s=0.2, alpha=0.2, c="olive", label='_nolegend_')
    plt.scatter(trajs[0][-1, :n, 0], trajs[0][-1, :n, 1], s=20, alpha=0.8, c="blue", marker='x')
    
    plt.scatter(trajs[1][0, :n, 0], trajs[1][0, :n, 1], s=15, alpha=0.8, c="red")
    plt.scatter(trajs[1][:, :n, 0], trajs[1][:, :n, 1], s=0.2, alpha=0.2, c="pink", label='_nolegend_')
    plt.scatter(trajs[1][-1, :n, 0], trajs[1][-1, :n, 1], s=20, alpha=0.8, c="red", marker='x')
    
    plt.legend([r"$\boldsymbol{x}|u_1$", r"$\widehat{\boldsymbol{y}}|u_1$", r"$\boldsymbol{x}|u_2$", r"$\widehat{\boldsymbol{y}}|u_2$"], fontsize=12)
    
    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    
    # Keep the grid lines
    # plt.grid(True)
    
    image = plt.gcf()
    plt.show()
    return image
    
def plot_multiple_trajectories_gradient(trajs, ticks=True):
    """Plot trajectories of some selected samples."""
    n = 2000
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create color gradient from blue to red
    num_steps = trajs[0].shape[0]
    colors = plt.cm.plasma(np.linspace(0, 1, num_steps))  # Using plasma colormap
    
    # Plot trajectory 1 with gradient
    for i in range(num_steps-1):
        ax.scatter(trajs[0][i, :n, 0], trajs[0][i, :n, 1], s=0.2, alpha=0.2, c=[colors[i]], label='_nolegend_')
    ax.scatter(trajs[0][0, :n, 0], trajs[0][0, :n, 1], s=15, alpha=0.8, c="blue")
    ax.scatter(trajs[0][-1, :n, 0], trajs[0][-1, :n, 1], s=20, alpha=0.8, c="blue", marker='x')
    
    # Plot trajectory 2 with gradient
    for i in range(num_steps-1):
        ax.scatter(trajs[1][i, :n, 0], trajs[1][i, :n, 1], s=0.2, alpha=0.2, c=[colors[i]], label='_nolegend_')
    ax.scatter(trajs[1][0, :n, 0], trajs[1][0, :n, 1], s=15, alpha=0.8, c="red")
    ax.scatter(trajs[1][-1, :n, 0], trajs[1][-1, :n, 1], s=20, alpha=0.8, c="red", marker='x')
    
    # Plot trajectory 3 if it exists
    if len(trajs) > 2:
        for i in range(num_steps-1):
            ax.scatter(trajs[2][i, :n, 0], trajs[2][i, :n, 1], s=0.2, alpha=0.2, c=[colors[i]], label='_nolegend_')
        ax.scatter(trajs[2][0, :n, 0], trajs[2][0, :n, 1], s=15, alpha=0.8, c="green")
        ax.scatter(trajs[2][-1, :n, 0], trajs[2][-1, :n, 1], s=20, alpha=0.8, c="green", marker='x')
        ax.legend([r"$\boldsymbol{x}|u_1$", r"$\widehat{\boldsymbol{y}}|u_1$", r"$\boldsymbol{x}|u_2$", r"$\widehat{\boldsymbol{y}}|u_2$", r"$\boldsymbol{x}|u_3$", r"$\widehat{\boldsymbol{y}}|u_3$"], fontsize=12)
    else:
        ax.legend([r"$\boldsymbol{x}|u_1$", r"$\widehat{\boldsymbol{y}}|u_1$", r"$\boldsymbol{x}|u_2$", r"$\widehat{\boldsymbol{y}}|u_2$"], fontsize=12)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Keep the grid lines
    # ax.grid(True)
    
    image = plt.gcf()
    plt.show()
    return image

def plot_pred_target(trajs, ticks=True):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    # for traj in trajs:
    plt.scatter(trajs[0][0, :n, 0], trajs[0][0, :n, 1], c="black", marker='.')
    plt.scatter(trajs[0][1, :n, 0], trajs[0][1, :n, 1], c="olive", marker='.')
    plt.scatter(trajs[0][2, :n, 0], trajs[0][2, :n, 1], c="black", marker='x')
    
    plt.scatter(trajs[1][0, :n, 0],  trajs[1][0, :n, 1],  c="red", marker='.')
    plt.scatter(trajs[1][1, :n, 0],  trajs[1][1, :n, 1], c="pink", marker='.')
    plt.scatter(trajs[1][2, :n, 0], trajs[1][2, :n, 1],  c="red", marker='x')

    plt.legend(["initial sample u=1", "translation u=1", "target sample u=1", "initial sample u=2", "translation u=2", "final sample u=2"])
    
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    image = plt.gcf()
    plt.show()
    return image


def plot_uncond_pred_target(trajs, ticks=True):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    # for traj in trajs:
    plt.scatter(trajs[1, :n, 0], trajs[1, :n, 1], c="red", alpha=0.5, marker='.')
    plt.scatter(trajs[2, :n, 0], trajs[2, :n, 1], c="black", alpha=0.5, marker='x')
    
    plt.legend(["translation", "target"])
    
    if not ticks:
        plt.xticks([])
        plt.yticks([])

    image = plt.gcf()
    plt.show()
    return image

def plot_multiple_trajectories_3d(trajs, ticks=True, x1=None):
    """Plot trajectories of some selected samples."""
    n = 2000
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # for traj in trajs:
    ax.scatter(trajs[0][0, :n, 0],  trajs[0][0, :n, 1],  trajs[0][0,  :n, 2],    s=15,  alpha=0.8, c="blue")
    ax.scatter(trajs[0][:, :n, 0],  trajs[0][:, :n, 1],  trajs[0][:,  :n, 2],    s=0.2, alpha=0.2, c="olive", label='_nolegend_')
    ax.scatter(trajs[0][-1, :n, 0], trajs[0][-1, :n, 1], trajs[0][-1, :n, 2],    s=20,   alpha=0.8,   c="blue", marker='x')

    ax.scatter(trajs[1][0, :n, 0],  trajs[1][0, :n, 1],  trajs[1][0, :n,  2],  s=15,  alpha=0.8, c="red")
    ax.scatter(trajs[1][:, :n, 0],  trajs[1][:, :n, 1],  trajs[1][:, :n,  2],  s=0.2, alpha=0.2, c="pink", label='_nolegend_')
    ax.scatter(trajs[1][-1, :n, 0], trajs[1][-1, :n, 1], trajs[1][-1, :n, 2],  s=20,   alpha=0.8,   c="red", marker='x')
    
    if x1 is not None:
        ax.scatter(x1[0][:n, 0],  x1[0][:n, 1],  x1[0][:n,  2],  s=20,  alpha=0.8, c="blue", marker='x')
        ax.scatter(x1[1][:n, 0],  x1[1][:n, 1],  x1[1][:n,  2],  s=20,  alpha=0.8, c="red", marker='x')
        ax.legend([r"$\boldsymbol{x}|u_1$", r"$\widehat{\boldsymbol{y}}|u_1$", r"$\boldsymbol{x}|u_2$", r"$\widehat{\boldsymbol{y}}|u_2$", r"$\boldsymbol{y}|u_1$", r"$\boldsymbol{y}|u_2$"], fontsize=12)
    else:
        ax.legend([r"$\boldsymbol{x}|u_1$", r"$\widehat{\boldsymbol{y}}|u_1$", r"$\boldsymbol{x}|u_2$", r"$\widehat{\boldsymbol{y}}|u_2$"], fontsize=12)

    # Remove ticks but keep lines
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Keep the grid lines for 3D effect
    ax.grid(True)
    ax.view_init(elev=90, azim=0)
    
    image = plt.gcf()
    plt.show()
    
    return image



def plot_multiple_trajectories_3d_gradient(trajs, ticks=True, x1=None):
    """Plot trajectories of some selected samples."""
    n = 2000
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create color gradients from blue to red
    n_steps = len(trajs[0])
    colors_1 = plt.cm.RdBu(np.linspace(0, 1, n_steps))
    colors_2 = plt.cm.RdBu_r(np.linspace(0, 1, n_steps))

    # Plot first trajectory with blue gradient
    ax.scatter(trajs[0][0, :n, 0], trajs[0][0, :n, 1], trajs[0][0, :n, 2], s=15, alpha=0.8, c=[colors_1[0]])
    for i in range(n_steps):
        ax.scatter(trajs[0][i, :n, 0], trajs[0][i, :n, 1], trajs[0][i, :n, 2], s=0.2, alpha=0.2, c=[colors_1[i]], label='_nolegend_')
    ax.scatter(trajs[0][-1, :n, 0], trajs[0][-1, :n, 1], trajs[0][-1, :n, 2], s=20, alpha=0.8, c=[colors_1[-1]], marker='x')

    # Plot second trajectory with red gradient  
    ax.scatter(trajs[1][0, :n, 0], trajs[1][0, :n, 1], trajs[1][0, :n, 2], s=15, alpha=0.8, c=[colors_2[0]])
    for i in range(n_steps):
        ax.scatter(trajs[1][i, :n, 0], trajs[1][i, :n, 1], trajs[1][i, :n, 2], s=0.2, alpha=0.2, c=[colors_2[i]], label='_nolegend_')
    ax.scatter(trajs[1][-1, :n, 0], trajs[1][-1, :n, 1], trajs[1][-1, :n, 2], s=20, alpha=0.8, c=[colors_2[-1]], marker='x')
    
    if x1 is not None:
        ax.scatter(x1[0][:n, 0], x1[0][:n, 1], x1[0][:n, 2], s=20, alpha=0.8, c="blue", marker='x')
        ax.scatter(x1[1][:n, 0], x1[1][:n, 1], x1[1][:n, 2], s=20, alpha=0.8, c="red", marker='x')
        ax.legend([r"$\boldsymbol{x}|u_1$", r"$\widehat{\boldsymbol{y}}|u_1$", r"$\boldsymbol{x}|u_2$", r"$\widehat{\boldsymbol{y}}|u_2$", r"$\boldsymbol{y}|u_1$", r"$\boldsymbol{y}|u_2$"], fontsize=12)
    else:
        ax.legend([r"$\boldsymbol{x}|u_1$", r"$\widehat{\boldsymbol{y}}|u_1$", r"$\boldsymbol{x}|u_2$", r"$\widehat{\boldsymbol{y}}|u_2$"], fontsize=12)

    # Remove ticks but keep lines
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    

    # Set the view angle from top (90 degree elevation)
    ax.view_init(elev=90, azim=0)
    # Keep the grid lines for 3D effect
    ax.grid(True)
    
    image = plt.gcf()
    plt.show()
    
    return image

