import ml_collections


def get_interpolant_config(config):
    # model
    config.interpolant_model = interpolant_model = ml_collections.ConfigDict()
    interpolant_model.type = "unet"

    interpolant_model.hidden_size = 64
    interpolant_model.dim_mults = [2, 2, 2]
    interpolant_model.num_res_blocks = 2
    interpolant_model.heads = 1
    interpolant_model.dim_head = 32
    interpolant_model.attention_resolution = [8]    
    interpolant_model.dropout = 0.1
    interpolant_model.biggan_sample = False
    interpolant_model.use_vae = True
    interpolant_model.input_shape = [4, 32, 32]


    # optimization
    config.interpolant_optim = interpolant_optim = ml_collections.ConfigDict()
    interpolant_optim.weight_decay = 0
    interpolant_optim.optimizer = "adam"
    interpolant_optim.learning_rate = 1e-4
    interpolant_optim.beta_one = 0.9
    interpolant_optim.beta_two = 0.999
    interpolant_optim.eps = 1e-8
    interpolant_optim.grad_clip = 1.0
    interpolant_optim.warmup = 0.0
    interpolant_optim.schedule = "constant"
    interpolant_optim.ema_decay = 0.9999

    return config
