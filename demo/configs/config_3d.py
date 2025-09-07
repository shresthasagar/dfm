from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    # Experiment
    cfg.dim = 3
    cfg.method = 'dfm'  # dfm, fm, fm_ot (overridden by CLI)
    cfg.max_iter = 4000
    cfg.batch_size = 512
    cfg.test_batch_size = 1000
    cfg.log_every = 1000

    # Data
    cfg.gaussian_center_type = '2_gaussians'
    cfg.gaussian_var = 1.0

    # OT
    cfg.ot_method = 'exact'

    # Model - unified v(t, x)
    cfg.v_net_width = 64
    cfg.v_hidden_layers = 2

    # Interpolant network (only used for dfm)
    cfg.inter_net_width = 64
    cfg.inter_hidden_layers = 2

    # Loss weights and schedules
    cfg.lambda_v = 0.01
    cfg.spatial_sigma = 1.0
    cfg.temporal_sigma = 0.1
    cfg.dfm_warmup_iters = 2000

    # Optimization
    cfg.vt_lr = 1e-3
    cfg.inter_lr = 1e-4
    cfg.grad_clip = 1.0

    # ODE
    cfg.traj_steps = 100
    cfg.ode_sensitivity = 'adjoint'
    cfg.ode_atol = 1e-4
    cfg.ode_rtol = 1e-4

    # Logging
    cfg.use_wandb = True
    cfg.wandb_project = 'flow_udt'

    return cfg


