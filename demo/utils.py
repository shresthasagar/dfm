from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
import matplotlib.cm as cm

# Optional/extra deps
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover
    go = None
    pio = None
    _PLOTLY_AVAILABLE = False

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.generate_data import *


class LinearInterpolant(nn.Module):
    """Linear interpolation between two points.

    Implements \(x(t) = (1 - t)\,x_0 + t\,x_1\). The time derivative is
    constant and equal to \(x_1 - x_0\).
    """
    def __init__(self, out_dim=None):
        super().__init__()
        # No parameters needed for linear interpolation
        pass
        
    def forward(self, x0, x1, t):
        """Compute linear interpolation between x0 and x1 at time t.
        
        Args:
            x0: Starting point tensor of shape [batch, dim]
            x1: Ending point tensor of shape [batch, dim]
            t: Time points tensor in [0, 1] of shape [batch] or [batch, 1]
            
        Returns:
            Interpolated points at time t with shape [batch, dim]
        """
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        return t * x1 + (1 - t) * x0
        
    def gradient(self, x0, x1, t):
        """Compute time derivative of the linear interpolant.
        
        Args:
            x0: Starting point tensor of shape [batch, dim]
            x1: Ending point tensor of shape [batch, dim]
            t: Time points tensor (unused for linear interpolant)
            
        Returns:
            Constant vector field x1 - x0 with shape [batch, dim]
        """
        return x1 - x0

class ParametricInterpolant(nn.Module):
    """Learnable non-linear interpolation between two points.

    Uses an MLP to learn a correction term to the linear interpolation,
    resulting in \(x(t) = (1-t)\,x_0 + t\,x_1 + t(1-t)\,\phi_\theta(t, x_0, x_1)\).
    """
    def __init__(self, out_dim=2, net_width=64, hidden_layers=2):
        """
        Args:
            out_dim: Dimension of output space
            net_width: Width of MLP hidden layers
            hidden_layers: Number of hidden layers in MLP
        """
        super().__init__()
        self.mlp = MLP(dim=(2*out_dim), out_dim=out_dim, time_varying=True, 
                      w=net_width, hidden_layers=hidden_layers)

    def forward(self, x0, x1, t, return_phi=False):
        """Compute non-linear interpolation between x0 and x1.
        
        Args:
            x0: Starting point tensor of shape [batch, dim]
            x1: Ending point tensor of shape [batch, dim]
            t: Time points tensor in [0, 1] of shape [batch] or [batch, 1]
            return_phi: If True, also return the learned correction term phi
            
        Returns:
            Interpolated points of shape [batch, dim]. If return_phi=True,
            returns tuple (interpolation, phi) where phi has shape [batch, dim]
        """
        t = t.reshape(-1, *([1] * (x0.dim() - 1)))
        phi = self.mlp(t, torch.cat([x0, x1], dim=-1))
        interpolation = t * x1 + (1-t) * x0 + t * (1-t) * phi
        
        if return_phi:
            return interpolation, phi
        return interpolation

    def gradient(self, x0, x1, t):
        """Compute time derivative of the parametric interpolant via autograd.
        
        Args:
            x0: Starting point tensor of shape [batch, dim]
            x1: Ending point tensor of shape [batch, dim]
            t: Time points tensor in [0, 1] of shape [batch] or [batch, 1]
            
        Returns:
            Time derivative tensor of shape [batch, dim]
        """
        t.requires_grad = True
        y = self.forward(x0, x1, t)

        grads = []
        for i in range(y.shape[-1]):
            grads.append(torch.autograd.grad(y[:,i].sum(), t, create_graph=True)[0])

        return torch.cat([grad.unsqueeze(-1) for grad in grads], dim=-1)

def get_cond_indices(y, u=0, type='easy'):
    """Get indices for conditional sampling from Gaussian mixture components.
    
    Args:
        y: Array of condition labels (cluster indices)
        u: Which conditional subset to sample from (0 or 1)
        type: Difficulty level of the conditional split ('easy', 'medium', 'hard')
        
    Returns:
        List of indices where condition labels match the specified subset
    """
    clusters = {
        'easy': [0,5,6,7],
        'medium': [0,1,2,7], 
        'hard': [0,2,4,6]
    }
    assert type in clusters, f'incorrect type {type}'
    
    if u == 1:
        indices = [i for i in range(len(y)) if y[i] in clusters[type]]
    else:
        indices = [i for i in range(len(y)) if y[i] not in clusters[type]]
    return indices

def get_two_gaussians(dim, batch_size=1000, var=1.0, center_type='2_gaussians', 
                     flow_type='reflection_origin'):
    """Generate paired samples from two Gaussian mixtures with conditional labels.
    
    Args:
        dim: Dimension of the data points
        batch_size: Total number of samples to generate
        var: Variance of the Gaussian components
        center_type: Type of Gaussian center configuration
        flow_type: Type of flow transformation between distributions
        
    Returns:
        Tuple of four tensors (x0_u0, x0_u1, x1_u0, x1_u1) where each tensor
        contains samples for source (x0) and target (x1) distributions split
        by condition u ∈ {0, 1}. All tensors have equal length.
    """
    data = GMMData(dim=dim, num_components=2, scale=5.0, var=var, 
                  cond=True, center_type=center_type, flow_type=flow_type)
    x0, x1, y0, y1 = data.get_data(n_samples=batch_size)
    
    # Split samples by condition
    x0_u0 = x0[y0 == 0]
    x0_u1 = x0[y0 == 1]
    x1_u0 = x1[y1 == 0]
    x1_u1 = x1[y1 == 1]

    # Ensure equal number of samples per condition
    min_samples = min(len(x0_u0), len(x1_u0), len(x0_u1), len(x1_u1))
    x0_u0 = x0_u0[:min_samples]
    x0_u1 = x0_u1[:min_samples]
    x1_u0 = x1_u0[:min_samples]
    x1_u1 = x1_u1[:min_samples]

    return x0_u0, x0_u1, x1_u0, x1_u1


def create_2d_trajectory_animation(trajs, save_path, filename="trajectory_animation_2d.gif", 
                                 n_points=100, fps=15, interval=150, pause_frames=30, 
                                 show_colorbar=True, square_aspect=True):
    """Create animated GIF of 2D trajectory evolution.
    
    Args:
        trajs: List of trajectory tensors [traj1, traj2] where each has shape (num_steps, num_samples, 2)
        save_path: Directory path to save the animation
        filename: Name of the output GIF file
        n_points: Number of points to animate per trajectory
        fps: Frames per second for the animation
        interval: Interval between frames in milliseconds
        pause_frames: Number of frames to pause at the end before looping
        show_colorbar: Whether to show the time progression colorbar
        square_aspect: Whether to use square aspect ratio
        
    Returns:
        Path to the saved animation file
    """
    def _to_np(t):
        return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    
    # Convert to numpy arrays
    trajs_np = [_to_np(traj) for traj in trajs]
    num_steps = trajs_np[0].shape[0]
    
    # Subsample points for performance
    trajs_sub = [traj[:, :n_points] for traj in trajs_np]
    
    # Set up the figure and axis with square aspect ratio and minimal margins
    if square_aspect:
        fig, ax = plt.subplots(figsize=(6, 6))  # Square figure
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Minimize margins around the plot
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
    
    # Calculate plot limits with very minimal margin
    all_x = np.concatenate([traj[:, :, 0].flatten() for traj in trajs_sub])
    all_y = np.concatenate([traj[:, :, 1].flatten() for traj in trajs_sub])
    margin = 0.01 * max(np.ptp(all_x), np.ptp(all_y))  # Very reduced margin
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Set square aspect ratio for the plot
    if square_aspect:
        ax.set_aspect('equal', adjustable='box')
    
    # Initialize trajectory line collections for rainbow effect
    traj1_lines = []
    traj2_lines = []
    
    # Start and end point plots
    start1_points = ax.scatter([], [], s=40, c='blue', marker='o', alpha=0.5, label='x|u_1 start')
    end1_points = ax.scatter([], [], s=50, c='blue', marker='x', alpha=0.9, label='x|u_1 current')
    start2_points = ax.scatter([], [], s=40, c='red', marker='o', alpha=0.5, label='x|u_2 start')
    end2_points = ax.scatter([], [], s=50, c='red', marker='x', alpha=0.9, label='x|u_2 current')
    
    # ax.set_title('Trajectory Evolution Animation', fontsize=12, fontweight='bold', pad=5)
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove axis spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add time indicator
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add color bar to show time progression (optional)
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Time Progression', rotation=270, labelpad=15)
    
    def animate(frame):
        # Clear previous trajectory lines
        for line in traj1_lines + traj2_lines:
            line.remove()
        traj1_lines.clear()
        traj2_lines.clear()
        
        # Determine current frame relative to animation (excluding pause frames)
        if frame < num_steps:
            current_frame = frame
            time_progress = frame / num_steps
        else:
            # During pause frames, show the final state
            current_frame = num_steps - 1
            time_progress = 1.0
        
        # Draw rainbow trajectory lines up to current frame
        if current_frame > 0:
            # Create color map for ALL time steps (fixed colors based on time)
            colors = cm.rainbow(np.linspace(0, 1, num_steps))
            
            # Sample fewer trajectories for performance (every nth sample)
            n_sample_lines = min(20, trajs_sub[0].shape[1])  # Much fewer lines
            sample_indices = np.linspace(0, trajs_sub[0].shape[1]-1, n_sample_lines, dtype=int)
            
            # Determine how many trajectory segments to draw
            max_t = min(current_frame, num_steps - 1)
            
            # Draw trajectory 1 with rainbow colors (only up to current frame)
            for t in range(max_t):
                # Connect consecutive time points with colors based on actual time t
                for i in sample_indices:  # Only sample subset of trajectories
                    x_vals = [trajs_sub[0][t, i, 0], trajs_sub[0][t+1, i, 0]]
                    y_vals = [trajs_sub[0][t, i, 1], trajs_sub[0][t+1, i, 1]]
                    # Use color based on the actual time step t, not animation frame
                    line, = ax.plot(x_vals, y_vals, color=colors[t], linewidth=0.6, alpha=0.7)
                    traj1_lines.append(line)
            
            # Draw trajectory 2 with rainbow colors (only up to current frame)
            for t in range(max_t):
                # Connect consecutive time points with colors based on actual time t
                for i in sample_indices:  # Only sample subset of trajectories
                    x_vals = [trajs_sub[1][t, i, 0], trajs_sub[1][t+1, i, 0]]
                    y_vals = [trajs_sub[1][t, i, 1], trajs_sub[1][t+1, i, 1]]
                    # Use color based on the actual time step t, not animation frame
                    line, = ax.plot(x_vals, y_vals, color=colors[t], linewidth=0.6, alpha=0.7)
                    traj2_lines.append(line)
        
        # Update start points (always visible)
        start1_points.set_offsets(trajs_sub[0][0, :, :2])
        start2_points.set_offsets(trajs_sub[1][0, :, :2])
        
        # Update current positions
        if current_frame < num_steps:
            end1_points.set_offsets(trajs_sub[0][current_frame, :, :2])
            end2_points.set_offsets(trajs_sub[1][current_frame, :, :2])
        
        # Update time indicator
        if frame < num_steps:
            time_text.set_text(f'Time: {time_progress:.2f}')
        else:
            time_text.set_text(f'Time: {time_progress:.2f} (Complete)')
        
        return traj1_lines + traj2_lines + [start1_points, end1_points, start2_points, end2_points, time_text]
    
    # Create animation (disable blit for dynamic line creation)
    total_frames = num_steps + pause_frames
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=interval, blit=False, repeat=True)
    
    # Save animation with tight bounding box
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, filename)
    print(f"Creating 2D animation with {num_steps} frames... (this may take a moment)")
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close()
    
    return output_path


def create_3d_trajectory_animation(trajs, save_path, filename="trajectory_animation_3d.gif",
                                 n_points=100, fps=12, interval=200, elev=30, azim_range=(0, 60), pause_frames=36,
                                 show_colorbar=True, square_aspect=True):
    """Create animated GIF of 3D trajectory evolution with rotation.
    
    Args:
        trajs: List of trajectory tensors [traj1, traj2] where each has shape (num_steps, num_samples, 3)
        save_path: Directory path to save the animation
        filename: Name of the output GIF file
        n_points: Number of points to animate per trajectory
        fps: Frames per second for the animation
        interval: Interval between frames in milliseconds
        elev: Elevation angle for 3D view
        azim_range: Tuple of (start_azim, end_azim) for rotation
        pause_frames: Number of frames to pause at the end before looping
        show_colorbar: Whether to show the time progression colorbar (not used in 3D)
        square_aspect: Whether to use square aspect ratio
        
    Returns:
        Path to the saved animation file
    """
    def _to_np(t):
        return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    
    # Convert to numpy arrays
    trajs_np = [_to_np(traj) for traj in trajs]
    num_steps = trajs_np[0].shape[0]
    
    # Subsample points for performance
    trajs_sub = [traj[:, :n_points] for traj in trajs_np]
    
    # Set up the figure and axis with square aspect ratio and minimal margins
    if square_aspect:
        fig = plt.figure(figsize=(8, 8))  # Square figure
    else:
        fig = plt.figure(figsize=(10, 8))
    
    # Minimize margins around the 3D plot
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate plot limits with very minimal margin
    all_x = np.concatenate([traj[:, :, 0].flatten() for traj in trajs_sub])
    all_y = np.concatenate([traj[:, :, 1].flatten() for traj in trajs_sub])
    all_z = np.concatenate([traj[:, :, 2].flatten() for traj in trajs_sub])
    
    margin = 0.01 * max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z))  # Very reduced margin
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_zlim(all_z.min() - margin, all_z.max() + margin)
    
    # Total frames: continuous rotation with trajectory building until t=1.0, then static
    extra_rotation_frames = 48  # Continue rotating for ~2-3 seconds after trajectory completes
    total_frames = num_steps + extra_rotation_frames
    
    def animate(frame):
        ax.clear()
        
        # Set limits again (cleared by ax.clear())
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax.set_zlim(all_z.min() - margin, all_z.max() + margin)
        
        # Single continuous phase: rotation throughout, trajectory builds until t=1.0 then stops
        
        # Calculate continuous rotation (total range over all frames)
        total_rotation = (azim_range[1] - azim_range[0]) + 60  # Initial range + extra rotation
        rotation_per_frame = total_rotation / (total_frames - 1)
        azim = azim_range[0] + frame * rotation_per_frame
        
        # Determine trajectory progress (stops at t=1.0)
        current_step = min(frame + 1, num_steps)  # Clamp at num_steps
        trajectory_complete = frame >= num_steps - 1
        
        # Draw rainbow trajectory lines up to current frame (or full if complete)
        if current_step > 0:
            # Create color map for ALL time steps (fixed colors based on time)
            colors = cm.rainbow(np.linspace(0, 1, num_steps))
            
            # Sample fewer points for performance
            n_sample_lines = min(20, trajs_sub[0].shape[1])
            indices = np.linspace(0, trajs_sub[0].shape[1]-1, n_sample_lines, dtype=int)
            
            # Determine how many trajectory segments to draw
            max_t = min(current_step - 1, num_steps - 2) if not trajectory_complete else num_steps - 2
            
            # Use different subsampling depending on whether trajectory is complete
            if trajectory_complete:
                # More detailed rendering when trajectory is complete and just rotating
                time_subsample = max(1, num_steps//10)
                draw_range = list(range(0, num_steps - time_subsample, time_subsample))
            else:
                # Standard rendering during trajectory evolution
                time_subsample = 1
                draw_range = list(range(max(0, max_t + 1)))
            
            # Draw trajectory 1 with rainbow colors
            for t in draw_range:
                if t + time_subsample < num_steps:
                    for i in indices:
                        x_vals = [trajs_sub[0][t, i, 0], trajs_sub[0][t+time_subsample, i, 0]]
                        y_vals = [trajs_sub[0][t, i, 1], trajs_sub[0][t+time_subsample, i, 1]]
                        z_vals = [trajs_sub[0][t, i, 2], trajs_sub[0][t+time_subsample, i, 2]]
                        ax.plot(x_vals, y_vals, z_vals, color=colors[t], linewidth=0.6, alpha=0.7)
            
            # Draw trajectory 2 with rainbow colors
            for t in draw_range:
                if t + time_subsample < num_steps:
                    for i in indices:
                        x_vals = [trajs_sub[1][t, i, 0], trajs_sub[1][t+time_subsample, i, 0]]
                        y_vals = [trajs_sub[1][t, i, 1], trajs_sub[1][t+time_subsample, i, 1]]
                        z_vals = [trajs_sub[1][t, i, 2], trajs_sub[1][t+time_subsample, i, 2]]
                        ax.plot(x_vals, y_vals, z_vals, color=colors[t], linewidth=0.6, alpha=0.7)
        
        # Highlight start points
        ax.scatter(trajs_sub[0][0, :, 0], trajs_sub[0][0, :, 1], trajs_sub[0][0, :, 2], 
                  s=40, alpha=0.5, c='blue', marker='o')
        ax.scatter(trajs_sub[1][0, :, 0], trajs_sub[1][0, :, 1], trajs_sub[1][0, :, 2], 
                  s=40, alpha=0.5, c='red', marker='o')
        
        # Highlight current or final positions
        if trajectory_complete:
            # Show final positions when trajectory is complete
            ax.scatter(trajs_sub[0][-1, :, 0], trajs_sub[0][-1, :, 1], trajs_sub[0][-1, :, 2], 
                      s=50, alpha=0.9, c='blue', marker='x')
            ax.scatter(trajs_sub[1][-1, :, 0], trajs_sub[1][-1, :, 1], trajs_sub[1][-1, :, 2], 
                      s=50, alpha=0.9, c='red', marker='x')
            ax.set_title('3D Trajectory - Final Result', fontsize=11, fontweight='bold', pad=5)
        else:
            # Show current positions during evolution
            if current_step < num_steps:
                ax.scatter(trajs_sub[0][current_step, :, 0], trajs_sub[0][current_step, :, 1], trajs_sub[0][current_step, :, 2], 
                          s=50, alpha=0.9, c='blue', marker='x')
                ax.scatter(trajs_sub[1][current_step, :, 0], trajs_sub[1][current_step, :, 1], trajs_sub[1][current_step, :, 2], 
                          s=50, alpha=0.9, c='red', marker='x')
            ax.set_title(f't={current_step/num_steps:.2f}', fontsize=11, fontweight='bold', pad=5)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=elev, azim=azim)
        # Make 3D panes less prominent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray') 
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=interval, repeat=True)
    
    # Save animation with tight bounding box
    os.makedirs(save_path, exist_ok=True)
    output_path = os.path.join(save_path, filename)
    print(f"Creating 3D animation with {total_frames} frames... (this may take several minutes)")
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close()
    
    return output_path


def save_3d_rotation_gif(trajs_np, save_path, elev=30, n_frames=36, fps=10):
    """Create a rotating GIF of 3D trajectories.
    
    Args:
        trajs_np: List of numpy arrays with trajectory data
        save_path: Path where to save the GIF
        elev: Elevation angle for 3D view
        n_frames: Number of rotation frames
        fps: Frames per second for the GIF
    """
    if imageio is None:
        print("imageio not available, skipping rotation GIF creation")
        return
        
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    n = min(1000, trajs_np[0].shape[1])  # Limit points for performance
    
    # Plot static trajectory data
    ax.scatter(trajs_np[0][0, :n, 0], trajs_np[0][0, :n, 1], trajs_np[0][0, :n, 2], s=15, alpha=0.8, c="blue")
    ax.scatter(trajs_np[0][:, :n, 0], trajs_np[0][:, :n, 1], trajs_np[0][:, :n, 2], s=0.5, alpha=0.3, c="olive")
    ax.scatter(trajs_np[0][-1, :n, 0], trajs_np[0][-1, :n, 1], trajs_np[0][-1, :n, 2], s=20, alpha=0.8, c="blue", marker='x')

    ax.scatter(trajs_np[1][0, :n, 0], trajs_np[1][0, :n, 1], trajs_np[1][0, :n, 2], s=15, alpha=0.8, c="red")
    ax.scatter(trajs_np[1][:, :n, 0], trajs_np[1][:, :n, 1], trajs_np[1][:, :n, 2], s=0.5, alpha=0.3, c="pink")
    ax.scatter(trajs_np[1][-1, :n, 0], trajs_np[1][-1, :n, 1], trajs_np[1][-1, :n, 2], s=20, alpha=0.8, c="red", marker='x')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True, alpha=0.3)
    
    # Create frames for rotation
    frames = []
    for i in range(n_frames):
        azim = 360 * i / n_frames
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw()
        
        # Convert to image
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(buf)
    
    # Save as GIF
    imageio.mimsave(str(save_path), frames, fps=fps, loop=0)
    plt.close()


def _build_plotly_3d_animation(trajs_np, save_html_path, title="3D Trajectories"):
    """Build interactive 3D animation using Plotly.
    
    Args:
        trajs_np: List of numpy arrays with trajectory data
        save_html_path: Path to save the HTML file
        title: Title for the plot
    """
    if not _PLOTLY_AVAILABLE:
        print("Plotly not available, skipping interactive HTML animation")
        return
        
    n = min(1000, trajs_np[0].shape[1])  # Limit points for performance
    num_steps = trajs_np[0].shape[0]
    
    frames = []
    
    for step in range(0, num_steps, max(1, num_steps//50)):  # Create frames for animation
        frame_data = []
        
        # Trajectory 1
        frame_data.append(go.Scatter3d(
            x=trajs_np[0][step, :n, 0],
            y=trajs_np[0][step, :n, 1],
            z=trajs_np[0][step, :n, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.7),
            name=f'Traj 1 (t={step/num_steps:.2f})'
        ))
        
        # Trajectory 2
        frame_data.append(go.Scatter3d(
            x=trajs_np[1][step, :n, 0],
            y=trajs_np[1][step, :n, 1],
            z=trajs_np[1][step, :n, 2],
            mode='markers',
            marker=dict(size=3, color='red', opacity=0.7),
            name=f'Traj 2 (t={step/num_steps:.2f})'
        ))
        
        frames.append(go.Frame(data=frame_data, name=f'frame_{step}'))
    
    # Initial frame
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=trajs_np[0][0, :n, 0],
                y=trajs_np[0][0, :n, 1],
                z=trajs_np[0][0, :n, 2],
                mode='markers',
                marker=dict(size=3, color='blue', opacity=0.7),
                name='Trajectory 1'
            ),
            go.Scatter3d(
                x=trajs_np[1][0, :n, 0],
                y=trajs_np[1][0, :n, 1],
                z=trajs_np[1][0, :n, 2],
                mode='markers',
                marker=dict(size=3, color='red', opacity=0.7),
                name='Trajectory 2'
            )
        ],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 100}}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
            ]
        )]
    )
    
    # Save to HTML
    fig.write_html(str(save_html_path))


def plot_multiple_trajectories_3d(
    trajs,
    ticks=True,
    x1=None,
    save_dir=None,
    filename_prefix="trajectories_3d",
    save_interactive_html=True,
    save_rotation_gif=False,
    create_animation=False,
    rotation_frames=36,
    show_colorbar=True,
    square_aspect=True,
    show=True,
):
    """Plot 3D trajectories (static MPL) and optionally save interactive HTML and a rotating GIF.

    Args:
        trajs: `[traj1, traj2]`, each of shape `(num_steps, num_samples, 3)`.
        ticks: Show axis ticks if True.
        x1: Optional target samples tuple/list `(x1_u0, x1_u1)` for markers.
        save_dir: Directory to save outputs (created if needed). If None, nothing is saved.
        filename_prefix: Prefix for saved files.
        save_interactive_html: If True and Plotly is available, save an interactive HTML animation.
        save_rotation_gif: If True, also save a rotating GIF suitable for README embedding.
        create_animation: If True, create an evolution animation GIF.
        rotation_frames: Number of frames for the rotating GIF.
        show: If True, display the static MPL figure via `plt.show()`.

    Returns:
        Matplotlib figure handle for the static plot.
    """
    # Static Matplotlib figure (for quick viewing / logging)
    n = 2000  # cap plotted points
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    def _to_np(t):
        return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

    trajs_np = [
        _to_np(trajs[0]),
        _to_np(trajs[1]),
    ]

    ax.scatter(trajs_np[0][0, :n, 0], trajs_np[0][0, :n, 1], trajs_np[0][0, :n, 2], s=15, alpha=0.8, c="blue")
    ax.scatter(trajs_np[0][:, :n, 0], trajs_np[0][:, :n, 1], trajs_np[0][:, :n, 2], s=0.2, alpha=0.2, c="olive", label='_nolegend_')
    ax.scatter(trajs_np[0][-1, :n, 0], trajs_np[0][-1, :n, 1], trajs_np[0][-1, :n, 2], s=20, alpha=0.8, c="blue", marker='x')

    ax.scatter(trajs_np[1][0, :n, 0], trajs_np[1][0, :n, 1], trajs_np[1][0, :n, 2], s=15, alpha=0.8, c="red")
    ax.scatter(trajs_np[1][:, :n, 0], trajs_np[1][:, :n, 1], trajs_np[1][:, :n, 2], s=0.2, alpha=0.2, c="pink", label='_nolegend_')
    ax.scatter(trajs_np[1][-1, :n, 0], trajs_np[1][-1, :n, 1], trajs_np[1][-1, :n, 2], s=20, alpha=0.8, c="red", marker='x')

    if x1 is not None:
        x1_np = (_to_np(x1[0]), _to_np(x1[1]))
        ax.scatter(x1_np[0][:n, 0], x1_np[0][:n, 1], x1_np[0][:n, 2], s=20, alpha=0.8, c="blue", marker='x')
        ax.scatter(x1_np[1][:n, 0], x1_np[1][:n, 1], x1_np[1][:n, 2], s=20, alpha=0.8, c="red", marker='x')
        ax.legend(["x|u_1", "y_hat|u_1", "x|u_2", "y_hat|u_2", "y|u_1", "y|u_2"], fontsize=12)
    else:
        ax.legend(["x|u_1", "y_hat|u_1", "x|u_2", "y_hat|u_2"], fontsize=12)

    if not ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    ax.grid(True)
    ax.view_init(elev=30, azim=90)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{filename_prefix}_static.png", 
                   bbox_inches='tight', pad_inches=0.1, dpi=200)
        
        if save_interactive_html and _PLOTLY_AVAILABLE:
            _build_plotly_3d_animation(trajs_np, save_html_path=save_path / f"{filename_prefix}_interactive.html", title=filename_prefix)
            
        if save_rotation_gif and imageio is not None:
            save_3d_rotation_gif(trajs_np, save_path=save_path / f"{filename_prefix}_rotation.gif", elev=30, n_frames=rotation_frames)
            
        if create_animation:
            try:
                # Convert back to tensor format for animation function
                trajs_tensor = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in trajs]
                anim_path = create_3d_trajectory_animation(trajs_tensor, str(save_path), f"{filename_prefix}_animation.gif", 
                                                         pause_frames=36, show_colorbar=show_colorbar, 
                                                         square_aspect=square_aspect)
                print(f"3D animation saved to: {anim_path}")
            except Exception as e:
                print(f"Failed to create 3D animation: {e}")

    if show:
        plt.show()
    return fig


def plot_multiple_trajectories(trajs, ticks=True, save_dir=None, filename_prefix="trajectories_2d", 
                              create_animation=False, show=True, show_colorbar=True, square_aspect=True):
    """Plot 2D trajectories of samples from two distributions.
    
    Args:
        trajs: List of trajectory tensors [traj1, traj2] where each trajectory has shape (num_steps, num_samples, 2)
        ticks: Whether to show axis ticks (default: True)
        save_dir: Directory to save plots and animations (created if needed). If None, nothing is saved.
        filename_prefix: Prefix for saved files
        create_animation: If True, also create an animated GIF
        show: If True, display the plot via plt.show()
        show_colorbar: Whether to show the time progression colorbar
        square_aspect: Whether to use square aspect ratio
        
    Returns:
        matplotlib figure object
    """
    n = 2000  # Number of points to plot
    if square_aspect:
        fig = plt.figure(figsize=(6, 6))
    else:
        fig = plt.figure(figsize=(8, 6))

    # Plot first trajectory (blue)
    plt.scatter(trajs[0][0, :n, 0], trajs[0][0, :n, 1], s=15, alpha=0.8, c="blue")      # Initial points
    plt.scatter(trajs[0][:, :n, 0], trajs[0][:, :n, 1], s=0.2, alpha=0.2, c="olive", label='_nolegend_')  # Intermediate points
    plt.scatter(trajs[0][-1, :n, 0], trajs[0][-1, :n, 1], s=20, alpha=0.8, c="blue", marker='x')  # Final points
    
    # Plot second trajectory (red)
    plt.scatter(trajs[1][0, :n, 0], trajs[1][0, :n, 1], s=15, alpha=0.8, c="red")       # Initial points
    plt.scatter(trajs[1][:, :n, 0], trajs[1][:, :n, 1], s=0.2, alpha=0.2, c="pink", label='_nolegend_')   # Intermediate points
    plt.scatter(trajs[1][-1, :n, 0], trajs[1][-1, :n, 1], s=20, alpha=0.8, c="red", marker='x')   # Final points
    
    plt.legend(["x|u_1", "y_hat|u_1", "x|u_2", "y_hat|u_2"], fontsize=12)
    
    # Configure axes
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.title('2D Trajectory Visualization', fontsize=14, fontweight='bold')
    
    # Set square aspect ratio
    if square_aspect:
        plt.gca().set_aspect('equal', adjustable='box')
    
    # Save plot if directory specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{filename_prefix}_static.png"), 
                   bbox_inches='tight', pad_inches=0.1, dpi=200)
        
        # Create animation if requested
        if create_animation:
            try:
                anim_path = create_2d_trajectory_animation(trajs, save_dir, f"{filename_prefix}_animation.gif", 
                                                         pause_frames=30, show_colorbar=show_colorbar, 
                                                         square_aspect=square_aspect)
                print(f"2D animation saved to: {anim_path}")
            except Exception as e:
                print(f"Failed to create 2D animation: {e}")
    
    if show:
        plt.show()
    return fig
