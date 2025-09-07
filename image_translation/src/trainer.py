# Standard library imports
import os
from collections import deque
import copy
import logging

# Third-party imports
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchdyn.core import NeuralODE
import lpips

# Local imports
from torchcfm.utils import torch_wrapper
from diffusers import AutoencoderKL
from torchcfm.models.unet.unet import UNetModelWrapper, InterpolantModelWrapper
from utils.interpolants import CondInterpolant, LinearInterpolant
from torchcfm.models.stargan import StarGANInterpolant, Generator
from torchcfm.optimal_transport import OTPlanSampler

class Trainer(object):
    """
    Main trainer class that handles model training and evaluation.
    Implements DFM for image translation.
    """
    
    def __init__(self, config):
        # Initialize model components
        self.ddm = None
        self.v_model_ema = None  # EMA version of velocity model
        self.ddm_optimizer = None
        self.vae = None 
        self.interpolant = None
        self.interpolant_optimizer = None
        self.config = config
        self.fm_loss = None

        # Build the model architecture
        self.build_model()

        # Initialize loss tracking
        self.losses = {
            'fm_loss': deque(maxlen=self.config.training.print_freq),
            'collision_penalty': deque(maxlen=self.config.training.print_freq),
            'spatial_collision_penalty': deque(maxlen=self.config.training.print_freq),
            'temporal_collision_penalty': deque(maxlen=self.config.training.print_freq)
        }

        # Initialize LPIPS loss and optimal transport sampler
        self.lpips_loss = lpips.LPIPS(net='alex').cuda()
        self.ot_sampler = OTPlanSampler(method='exact')

    def build_model(self):
        """
        Builds and initializes all model components including:
        - Velocity model (UNet or StarGAN)
        - EMA model
        - Interpolant
        - VAE (optional)
        - Optimizers
        """
        # Build velocity model based on config type
        if self.config.model.type == "unet":
            self.v_model = UNetModelWrapper(
                dim=self.config.model.input_shape,
                num_res_blocks=self.config.model.num_res_blocks,
                num_channels=self.config.model.hidden_size,
                channel_mult=self.config.model.dim_mults,
                num_heads=self.config.model.heads,
                num_head_channels=self.config.model.dim_head,
                attention_resolutions=str(self.config.model.attention_resolution[0]),
                dropout=self.config.model.dropout,
                class_cond=False,
                use_checkpoint=False
            )
        elif self.config.model.type == "stargan":
            self.v_model = Generator(
                img_size=self.config.model.img_size, 
                max_conv_dim=self.config.model.max_conv_dim,
                embed_dim=self.config.model.embed_dim,
                num_classes=1
            )
            
        # Create EMA model
        self.v_model_ema = copy.deepcopy(self.v_model)    
        for param in self.v_model_ema.parameters():
            param.requires_grad = False   
        self.v_model_ema.eval()

        # Build interpolant if using parametrized version
        if self.config.use_parametrized_interpolant:
            interp_net = InterpolantModelWrapper(
                dim=(self.config.model.input_shape[0]*2, *self.config.interpolant_model.input_shape[1:]),
                num_res_blocks=self.config.interpolant_model.num_res_blocks,
                num_channels=self.config.interpolant_model.hidden_size,
                channel_mult=self.config.interpolant_model.dim_mults,
                num_heads=self.config.interpolant_model.heads,
                num_head_channels=self.config.interpolant_model.dim_head,
                attention_resolutions=str(self.config.interpolant_model.attention_resolution[0]),
                dropout=0,
                class_cond=True,
                out_channels=self.config.model.input_shape[0],
                use_checkpoint=False
            )
            self.interpolant = CondInterpolant(interp_net)
        else:
            self.interpolant = LinearInterpolant()

        # Load VAE if specified
        if self.config.model.use_vae:
            self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").eval()

        # Initialize optimizers
        self.ddm_optimizer = optim.AdamW(
            self.v_model.parameters(),
            lr=self.config.optim.learning_rate,
            betas=[self.config.optim.beta_one, self.config.optim.beta_two],
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )

        if self.config.use_parametrized_interpolant:
            self.interpolant_optimizer = optim.AdamW(
                self.interpolant.parameters(),
                lr=self.config.optim.learning_rate,
                betas=[self.config.optim.beta_one, self.config.optim.beta_two],
                eps=self.config.optim.eps,
                weight_decay=self.config.optim.weight_decay
            )

        # Move models to GPU
        self.v_model.cuda()
        self.v_model_ema.cuda()
        if self.config.use_parametrized_interpolant:
            self.interpolant.cuda()
        if self.config.model.use_vae:
            self.vae.cuda()

        # Uncomment Compile models for better performance
        # self.v_model = torch.compile(self.v_model)
        # if self.config.use_parametrized_interpolant:
        #     self.interpolant = torch.compile(self.interpolant)

    def merge_images_all(self, sources, targets, k=10):
        """Merges source and target images into a single grid for visualization"""
        _, _, h, w = sources.shape
        row = min(int(np.sqrt(len(sources))), 8)
        merged = np.zeros([3, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            if idx >= row*row:
                break
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
        return merged.transpose(1,2,0)

    def merge_images_all_plt(self, sources, targets, recons):
        """Creates a matplotlib figure with source, target and reconstruction images"""
        num_images, n_cols = sources.shape[0], 3
        sources, targets, recons = sources*0.5 + 0.5, targets*0.5 + 0.5, recons*0.5 + 0.5
        fig = plt.figure(figsize=(n_cols*10, (num_images)*3))
        grid = ImageGrid(fig, 111, nrows_ncols=(num_images, n_cols), axes_pad=0.1)
        fontsize = 15
        
        for i in range(num_images):
            grid[i*n_cols].imshow(np.transpose(sources[i],(1, 2, 0)))
            grid[i*n_cols+1].imshow(np.transpose(targets[i],(1, 2, 0)))
            grid[i*n_cols+2].imshow(np.transpose(recons[i],(1, 2, 0)))
            
            # Set titles for first row
            if i == 0:
                grid[i*n_cols].set_title('Source', fontsize=fontsize)
                grid[i*n_cols+1].set_title('Translation', fontsize=fontsize)
                grid[i*n_cols+2].set_title('Recons.', fontsize=fontsize)

        # Remove ticks
        for ax in grid:
            ax.set_xticks([])
            ax.set_yticks([])
            
        return fig

    def save_image_eval(self, test_domain1):
        """Generates and saves evaluation images"""
        fake_domain1 = self.sample(self.v_model_ema, test_domain1, labels=None)
        
        domain1, fake_domain1 = test_domain1.detach().cpu().numpy(), fake_domain1.clamp_(-1,1).detach().cpu().numpy()
        fig1 = self.merge_images_all(domain1, fake_domain1)
        return fig1
    
    def calculate_lpips_loss(self, x1, x2):
        """Calculates LPIPS perceptual loss between two images"""
        # Reshape from [N, 4, W, W] to [N*4, 1, W, W]
        N = x1.shape[0]
        x1 = x1.reshape(-1, 1, x1.shape[2], x1.shape[3])
        x2 = x2.reshape(-1, 1, x2.shape[2], x2.shape[3])
        
        # Repeat channel 3 times for LPIPS
        x1 = x1.repeat(1, 3, 1, 1)
        x2 = x2.repeat(1, 3, 1, 1)
        max_val = max(x1.abs().max(), x2.abs().max())
        return self.lpips_loss(x1/max_val, x2/max_val).reshape(N, -1).mean(dim=-1)

    def step(self, x0, x1, labels, iterations):
        """
        Performs one training step:
        1. Zero gradients
        2. Sample time points
        3. Apply optimal transport if enabled
        4. Calculate interpolation
        5. Compute velocity field
        6. Calculate losses (FM loss + collision penalty)
        7. Update models
        """
        # Zero gradients
        self.ddm_optimizer.zero_grad()
        if self.config.use_parametrized_interpolant and iterations > self.config.training.num_steps_no_interpolant:
            self.interpolant_optimizer.zero_grad()

        # Sample random time points
        t = torch.rand(x0.shape[0]).type_as(x0)
        
        # Handle single conditional case
        if self.config.data.num_conditionals == 1:
            labels = torch.zeros_like(labels)

        # Apply optimal transport sampling if enabled
        if self.config.training.use_ot:
            # Group data by labels
            unique_labels = torch.unique(labels)
            x0_groups = [x0[labels==l] for l in unique_labels]
            x1_groups = [x1[labels==l] for l in unique_labels]
            
            # Sample from OT plan for each group
            x0_ot = []
            x1_ot = []
            labels_ot = []
            for i, (x0_g, x1_g) in enumerate(zip(x0_groups, x1_groups)):
                x0_sampled, x1_sampled = self.ot_sampler.sample_plan(x0_g, x1_g, replace=True)
                x0_ot.append(x0_sampled)
                x1_ot.append(x1_sampled)
                labels_ot.append(torch.full((len(x0_sampled),), unique_labels[i], device=labels.device))
            
            # Combine resampled data
            x0 = torch.cat(x0_ot, dim=0)
            x1 = torch.cat(x1_ot, dim=0)
            labels = torch.cat(labels_ot, dim=0)

        # Handle interpolation labels
        interp_labels = labels
        if 'same_interp' in self.config.training:
            interp_labels = torch.zeros_like(interp_labels)
    
        # Get interpolation and phi
        xt, phi = self.interpolant.get_interp_and_phi(t, x0, x1, interp_labels)

        # Calculate ground truth velocity
        if self.config.use_parametrized_interpolant:
            with torch.no_grad():
                ut = self.interpolant.gradient(t.unsqueeze(-1), x0.unsqueeze(1), x1.unsqueeze(1), interp_labels.unsqueeze(1))
        else:
            with torch.no_grad():
                ut = self.interpolant.gradient(t, x0, x1)

        # Compute predicted velocity
        vt = self.v_model(t, xt.detach())
        fm_loss = torch.mean((vt - ut.detach()) ** 2) 

        # Initialize collision penalties
        collision_penalty = torch.tensor(0.0)
        spatial_collision_penalty = torch.tensor(0.0)
        temporal_collision_penalty = torch.tensor(0.0)

        # Compute collision penalty if enabled
        if self.config.training.collision_penalty_wt > 0:
            xts = [xt[labels==j] for j in range(self.config.data.num_conditionals)]
            ts = [t[labels==j] for j in range(self.config.data.num_conditionals)]
            min_size = min([len(xts[j]) for j in range(len(xts))])
            xts = [xts[j][:min_size] for j in range(len(xts))]
            ts = [ts[j][:min_size] for j in range(len(ts))]
            
            # Calculate penalties between all pairs
            for j in range(len(xts)-1):
                for k in range(j+1, len(xts)):
                    ts_j, ts_k = ts[j], ts[k]
                    if self.config.training.use_lpips_spatial:
                        xtj, xtk = xts[j], xts[k]
                        spatial_collision_penalty = self.calculate_lpips_loss(xtj, xtk)
                    else:
                        xtj, xtk = xts[j].view(xts[j].shape[0], -1), xts[k].view(xts[k].shape[0], -1)
                        spatial_collision_penalty = torch.norm(xtj - xtk, dim=-1)/np.sqrt(xts[j].shape[-1])
                    temporal_collision_penalty = (ts_j - ts_k)
                    
                    collision_penalty = collision_penalty + \
                        torch.mean(torch.exp( - spatial_collision_penalty**2 / (2 * (self.config.training.spatial_sigma**2)) \
                                            - temporal_collision_penalty**2 / (2 * (self.config.training.temporal_sigma**2))))

        # Compute total loss and backpropagate
        total_loss = fm_loss + \
                    self.config.training.collision_penalty_wt * collision_penalty
        total_loss.backward()
        
        # Check for NaN loss
        if torch.isnan(total_loss):
            # Debugging to see what is causing the NaN loss
            import IPython; IPython.embed(); exit();
        
        # Clip gradients if enabled
        if self.config.optim.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.v_model.parameters(), self.config.optim.grad_clip)
            if self.config.use_parametrized_interpolant:
                nn.utils.clip_grad_norm_(self.interpolant.parameters(), self.config.optim.grad_clip)

        # Update models
        self.ddm_optimizer.step()
        if self.config.use_parametrized_interpolant:
            self.interpolant_optimizer.step()

        # Update EMA model
        self.moving_average(self.v_model, self.v_model_ema, beta=self.config.optim.ema_decay)

        # Track losses
        self.losses['fm_loss'].append(fm_loss.item())
        self.losses['collision_penalty'].append(collision_penalty.item())
        self.losses['spatial_collision_penalty'].append(torch.mean(spatial_collision_penalty).item())
        self.losses['temporal_collision_penalty'].append(torch.mean(temporal_collision_penalty).item())
        
    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        """Updates EMA model parameters"""
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    @torch.no_grad()
    def sample(self, model, x0, labels=None, device='cuda', solver='dopri5'):
        """
        Samples from the model using Neural ODE integration
        Args:
            model: The velocity field model
            x0: Initial condition
            labels: Optional conditioning labels
            device: Device to run on
            solver: ODE solver type
        """
        if self.config.model.use_vae:
            self.vae.to(device)
            x0 = self.vae.encode(x0).latent_dist.mode()
            
        node = NeuralODE(
            torch_wrapper(model, labels),
            solver=solver, 
            sensitivity="adjoint", 
            atol=1e-4, 
            rtol=1e-4
        )

        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, 100),       
            )
            
        if self.config.model.use_vae:
            out = self.vae.decode(traj[-1]).sample
            self.vae.to(device)
            return out
        else:
            return traj[-1]

    def save_checkpoint(self, filename, iterations):
        """Saves model checkpoint"""
        params = {
            'v_model': self.v_model.state_dict(),
            'interpolant': self.interpolant.state_dict(),
            'ddm_optimizer': self.ddm_optimizer.state_dict(),
            'interpolant_optimizer': self.interpolant_optimizer.state_dict() if self.config.use_parametrized_interpolant else None,
            'v_model_ema': self.v_model_ema.state_dict(),
            'step': iterations
        }
        torch.save(params, filename)
        
    def load_checkpoint(self, checkpoint_path):
        """Loads model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.v_model.load_state_dict(checkpoint['v_model'])
            self.ddm_optimizer.load_state_dict(checkpoint['ddm_optimizer'])
            self.v_model_ema.load_state_dict(checkpoint['v_model_ema'])
            if self.config.use_parametrized_interpolant:
                self.interpolant.load_state_dict(checkpoint['interpolant'])
                self.interpolant_optimizer.load_state_dict(checkpoint['interpolant_optimizer'])
            return checkpoint['step']
        else:
            print(f'Loading Checkpoint Failed. Checkpoint {checkpoint_path} does not exist.')
                     
    def log_err_wandb(self, iterations):
        """Logs errors to Weights & Biases"""
        result = {}
        for key in self.losses:
            result[key] = np.mean(self.losses[key])
        wandb.log(result, step=iterations)

    def log_err_console(self, it):
        """Logs errors to console"""
        logging.info('iter: %d, fm_loss: %.4f, collision_penalty: %.4f' %  \
                    (it, np.mean(self.losses['fm_loss']), np.mean(self.losses['collision_penalty'])))
    
    def update_learning_rate(self, multiplier):
        """Updates learning rate for all optimizers"""
        for param_group in self.ddm_optimizer.param_groups:
            param_group['lr'] = self.config.optim.learning_rate * multiplier

    def get_current_learning_rate(self):
        """Returns current learning rate"""
        return self.ddm_optimizer.param_groups[0]['lr']
