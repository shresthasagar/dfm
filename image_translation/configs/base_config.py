import ml_collections


def get_base_config():
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.overfit_to_one_batch = False
    config.wandb_key = ""
    config.wandb_group = ""
    config.wandb_entity = ""
    config.wandb_run_id = ""
    config.wandb_resume_run = False

    # training
    config.training = training = ml_collections.ConfigDict()
    training.print_freq = 1000
    training.save_checkpoints = True
    training.preemption_ckpt = True
    training.ckpt_freq = 10000
    training.resume_ckpt = False
    training.resume_step = 0
    training.eval_only = False
    training.skip_training = False
    training.sample_freq = 10000
    training.use_ot = False


    config.eval = eval = ml_collections.ConfigDict()
    eval.compute_metrics = True
    eval.enable_fid = True
    eval.enable_path_lengths = True
    eval.enable_mse = False
    eval.checkpoint_metric = "fid"
    eval.save_samples = True
    eval.num_save_samples = 7
    eval.labelwise = True
    eval.checkpoint_step = 0

    config.use_parametrized_interpolant = False
    config.interpolant_type = 'linear'
    config.conditional_coupling = True
    config.training.load_pretrained_wts = ""
    config.training.num_steps_no_interpolant = 0
    config.training.eval_private_field = False

    return config
