# Import required libraries
from tqdm import trange
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
import torch
import numpy as np
import random
import time
from torchdyn.core import NeuralODE
from utils import *
try:
    from .configs import config_2d, config_3d  # when executed as a module
except Exception:
    from configs import config_2d, config_3d  # when executed as a script from the synthetic dir
import argparse
import os
from pathlib import Path

# Set up command line argument parser
parser = argparse.ArgumentParser(description='Flow Matching for Synthetic Data')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--method', type=str, default='dfm',
                    choices=['dfm', 'fm', 'fm_ot'],
                    help='training method: dfm (parametric interpolant), fm (linear interpolant), fm_ot (linear + OT)')
parser.add_argument('--config', type=str, default='2d', choices=['2d', '3d'],
                    help='which config to use (2d or 3d)')
parser.add_argument('--max_iter', type=int, default=None,
                    help='override maximum number of training iterations')
parser.add_argument('--batch_size', type=int, default=None,
                    help='override batch size')
parser.add_argument('--create_animations', action='store_true',
                    help='create animated GIFs in addition to static plots')
parser.add_argument('--results_dir', type=str, default='results',
                    help='directory to save results (default: results)')

args = parser.parse_args()

# Load configuration
cfg = config_2d.get_config() if args.config == '2d' else config_3d.get_config()

# Apply CLI overrides
if args.max_iter is not None:
    cfg.max_iter = args.max_iter
if args.batch_size is not None:
    cfg.batch_size = args.batch_size
cfg.method = args.method

# Set random seeds for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

params = cfg  # alias for readability

# Set method-specific parameters
if params.method == 'dfm':
    linear_inter = False
    use_ot = False
elif params.method == 'fm':
    linear_inter = True
    use_ot = False
elif params.method == 'fm_ot':
    linear_inter = True
    use_ot = True

run_name = f"{params.method}_dim{params.dim}"

# Setup results directory
results_dir = Path(args.results_dir)
results_dir.mkdir(exist_ok=True)
run_results_dir = results_dir / f"{run_name}_seed{args.seed}"
run_results_dir.mkdir(exist_ok=True)

print(f"Saving results to: {run_results_dir}")

# Training hyperparameters
dim = params.dim
max_iter = params.max_iter
batch_size = params.batch_size
gaussian_center_type = params.gaussian_center_type
gaussian_var = params.gaussian_var

# Initialize models
model_unified = MLP(dim=dim, time_varying=True, w=params.v_net_width, 
                   hidden_layers=params.v_hidden_layers)

ot_sampler = OTPlanSampler(method=params.ot_method)

# Initialize interpolants
if not linear_inter:
    inter = ParametricInterpolant(out_dim=dim, net_width=params.inter_net_width, 
                                 hidden_layers=params.inter_hidden_layers)
else:
    inter = LinearInterpolant()

# Initialize optimizers
optimizer_vt = torch.optim.Adam(model_unified.parameters(), lr=params.vt_lr)
if not linear_inter:
    optimizer_inter = torch.optim.Adam(inter.parameters(), lr=params.inter_lr)

# Initialize wandb logging
if getattr(params, 'use_wandb', False):
    import wandb
    wandb_config = params.to_dict() if hasattr(params, 'to_dict') else dict(params)
    wandb.init(project=params.wandb_project, name=run_name, config=wandb_config)

start = time.time()

# Training loop
for k in trange(max_iter):
    # Adjust lambda values based on iteration
    if params.method == 'dfm' and k < getattr(params, 'dfm_warmup_iters', 2000):
        lambda_fm = 0.0
        lambda_v_current = params['lambda_v']
    else:
        lambda_fm = 1.0
        lambda_v_current = 0.0

    # Zero gradients
    optimizer_vt.zero_grad()
    if not linear_inter:
        optimizer_inter.zero_grad()

    # Generate data
    x0_u0, x0_u1, x1_u0, x1_u1 = get_two_gaussians(dim, batch_size, var=gaussian_var, 
                                                   center_type=gaussian_center_type)

    # Generate random time points
    t0 = torch.rand(x0_u0.shape[0]).type_as(x0_u0)
    t1 = torch.rand(x0_u1.shape[0]).type_as(x0_u1)
    
    # Check for NaN in input data
    if torch.isnan(x0_u0).any() or torch.isnan(x1_u0).any() or \
       torch.isnan(x0_u1).any() or torch.isnan(x1_u1).any():
        print("NaN found in input data")
        continue

    # Apply optimal transport if specified
    if use_ot:
        x0_u0, x1_u0 = ot_sampler.sample_plan(x0_u0, x1_u0)
        x0_u1, x1_u1 = ot_sampler.sample_plan(x0_u1, x1_u1)

    # Compute interpolations and gradients
    xt1 = inter(x0_u0, x1_u0, t0)
    xt2 = inter(x0_u1, x1_u1, t1)

    ut1 = inter.gradient(x0_u0, x1_u0, t0).detach()
    ut2 = inter.gradient(x0_u1, x1_u1, t1).detach()

    vt1 = model_unified(t0[:, None], xt1.detach())
    vt2 = model_unified(t1[:, None], xt2.detach())

    # Compute losses
    fm1_loss = torch.mean((vt1 - ut1) ** 2) 
    fm2_loss = torch.mean((vt2 - ut2) ** 2)
    
    spatial_loss = torch.norm(xt1 - xt2, dim=-1)
    temporal_loss = torch.abs(t1 - t0) 

    # Compute collision penalty and total loss
    collision_penalty = torch.mean(torch.exp(-spatial_loss**2 / (2 * params['spatial_sigma']**2) - 
                                           temporal_loss**2 / (2 * params['temporal_sigma']**2)))
    loss = lambda_fm * (fm1_loss + fm2_loss) + lambda_v_current * collision_penalty

    # Backward pass and optimization
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model_unified.parameters(), max_norm=params.grad_clip)
    if not linear_inter:
        torch.nn.utils.clip_grad_norm_(inter.parameters(), max_norm=params.grad_clip)
    
    optimizer_vt.step()
    if not linear_inter:
        optimizer_inter.step()

    # Log metrics to wandb
    if getattr(params, 'use_wandb', False):
        wandb.log({
            'loss': loss.item(),
            'fm1_loss': fm1_loss.item(),
            'fm2_loss': fm2_loss.item(),
            'collision_penalty': collision_penalty.item(),
            'spatial_loss': spatial_loss.mean().item(),
            'temporal_loss': temporal_loss.mean().item()
        })

    # Periodic evaluation and visualization
    if (k) % params.log_every == 0:
        if getattr(params, 'use_wandb', False):
            wandb.watch(model_unified)
        end = time.time()
        print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
        start = end
        
        # Initialize Neural ODE
        node_unified = NeuralODE(
            torch_wrapper(model_unified), solver="dopri5", 
            sensitivity=params.ode_sensitivity, atol=params.ode_atol, rtol=params.ode_rtol
        )
        
        # Visualization
        with torch.no_grad():
            x0_u0, x0_u1, x1_u0, x1_u1 = get_two_gaussians(dim, batch_size, 
                                                          var=gaussian_var, 
                                                          center_type=gaussian_center_type)
            
            # Visualize interpolation trajectories
            t_span = torch.linspace(0, 1, params.traj_steps).type_as(x0_u0).unsqueeze(-1)
            traj1 = torch.stack([inter(x0_u0, x1_u0, t.repeat(x0_u0.shape[0], 1)) 
                               for t in t_span])
            traj2 = torch.stack([inter(x0_u1, x1_u1, t.repeat(x0_u1.shape[0], 1)) 
                               for t in t_span])
            trajs = [traj1, traj2]

            # Plot trajectories based on dimension
            if dim == 2:
                gcf = plot_multiple_trajectories(
                    [t.cpu().numpy() for t in trajs],
                    save_dir=str(run_results_dir),
                    filename_prefix=f"individual_iter{k}_2d",
                    create_animation=False,  # No animations during training
                    show=False
                )
            elif dim == 3:
                gcf = plot_multiple_trajectories_3d(
                    [t.cpu().numpy() for t in trajs],
                    save_dir=str(run_results_dir),
                    filename_prefix=f"individual_iter{k}_3d",
                    create_animation=False,  # No animations during training
                    save_interactive_html=False,  # No interactive plots during training
                    save_rotation_gif=False,  # No rotation GIFs during training
                    show=False
                )
            if getattr(params, 'use_wandb', False):
                wandb.log({'individual':wandb.Image(gcf)})

            # Visualize unified model trajectories
            traj1 = node_unified.trajectory(
                x0_u0,
                t_span=torch.linspace(0, 1, params.traj_steps),
            )
            traj2 = node_unified.trajectory(
                x0_u1,
                t_span=torch.linspace(0, 1, params.traj_steps),
            )
            trajs = [traj1, traj2]
            
            if dim == 2:
                gcf = plot_multiple_trajectories(
                    [t.cpu().numpy() for t in trajs],
                    save_dir=str(run_results_dir),
                    filename_prefix=f"unified_iter{k}_2d",
                    create_animation=False,  # No animations during training
                    show=False
                )
            elif dim == 3:
                gcf = plot_multiple_trajectories_3d(
                    [t.cpu().numpy() for t in trajs],
                    save_dir=str(run_results_dir),
                    filename_prefix=f"unified_iter{k}_3d",
                    create_animation=False,  # No animations during training
                    save_interactive_html=False,  # No interactive plots during training
                    save_rotation_gif=False,  # No rotation GIFs during training
                    show=False
                )
            
            if getattr(params, 'use_wandb', False):
                wandb.log({'unified':wandb.Image(gcf)})
            
# Evaluation on test data
test_batch_size = params.test_batch_size
x0_u0_test, x0_u1_test, x1_u0_test, x1_u1_test = get_two_gaussians(dim, test_batch_size, 
                                                                   var=gaussian_var, 
                                                                   center_type=gaussian_center_type)

# Compute test trajectories using unified model
with torch.no_grad():
    t_span = torch.linspace(0, 1, params.traj_steps)
    traj1_test = node_unified.trajectory(
        x0_u0_test,
        t_span=t_span,
    )
    traj2_test = node_unified.trajectory(
        x0_u1_test,
        t_span=t_span,
    )

    # Compute evaluation metrics
    te1 = torch.mean(torch.norm(-x0_u0_test - traj1_test[-1], dim=-1))
    te2 = torch.mean(torch.norm(-x0_u1_test - traj2_test[-1], dim=-1))
    te = (te1 + te2) / 2

    emd1 = wasserstein(x1_u0_test, traj1_test[-1], power=1)
    emd2 = wasserstein(x1_u1_test, traj2_test[-1], power=1)
    emd = (emd1 + emd2) / 2

    # Log final metrics
    if getattr(params, 'use_wandb', False):
        wandb.log({
            'translation_error': te.item(),
            'emd': emd
        })
    print(f"Translation Error: {te.item():.4f}")
    print(f"EMD: {emd:.4f}")

    # Create final visualizations with test data
    print(f"\nCreating final visualizations...")
    final_trajs = [traj1_test, traj2_test]
    
    # Only show colorbar for DFM method (for comparison plots)
    show_colorbar = (params.method == 'dfm')
    
    if dim == 2:
        final_gcf = plot_multiple_trajectories(
            [t.cpu().numpy() for t in final_trajs],
            save_dir=str(run_results_dir),
            filename_prefix="final_test_2d",
            create_animation=args.create_animations,
            show_colorbar=show_colorbar,
            square_aspect=True,
            show=False
        )
    elif dim == 3:
        final_gcf = plot_multiple_trajectories_3d(
            [t.cpu().numpy() for t in final_trajs],
            save_dir=str(run_results_dir),
            filename_prefix="final_test_3d",
            create_animation=args.create_animations,
            save_interactive_html=True,
            save_rotation_gif=True,
            show_colorbar=show_colorbar,
            square_aspect=True,
            show=False
        )
    
    # Save final metrics
    metrics_file = run_results_dir / "final_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Method: {params.method}\n")
        f.write(f"Dimension: {dim}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Translation Error: {te.item():.6f}\n")
        f.write(f"EMD: {emd:.6f}\n")
        f.write(f"Max Iterations: {max_iter}\n")
        f.write(f"Batch Size: {batch_size}\n")
    
    print(f"Final results saved to: {run_results_dir}")
    if args.create_animations:
        print(f"Animations created in: {run_results_dir}")

if getattr(params, 'use_wandb', False):
    wandb.finish()