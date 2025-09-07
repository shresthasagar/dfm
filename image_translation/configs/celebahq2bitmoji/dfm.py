# import ml_collections
from configs.base_ddm import get_ddm_config
from configs.celebahq2bitmoji.base_unet import get_unet_config
from configs.celebahq2bitmoji.base_celebahq2bimoji import get_celebahq2bimoji_config
from configs.celebahq2bitmoji.base_interpolant import get_interpolant_config

def get_config():
    config = get_ddm_config()
    config = get_unet_config(config)
    config = get_interpolant_config(config)
    config = get_celebahq2bimoji_config(config)

    config.name = "dfm"
    config.data.vae_name = "ddm_celebahq2bitmoji_gender"
    config.data.horizontal_flip = True
    config.data.attributes = ['Male', '~Male']
    config.data.num_conditionals = 2

    config.data.source_center_crop = False
    config.data.source_crop_ratio = 1.0
    config.data.target_center_crop = True
    config.data.target_crop_ratio = 0.8

    config.use_parametrized_interpolant = True
    
    config.training.same_interp = True
    config.training.sample_freq = 10_000
    config.training.num_steps = 100_000
    config.training.eval_freq = 10_000
    config.training.print_freq = 100
    config.training.max_samples = None

    config.training.resume_ckpt = False

    config.data.source_max_samples = 5000
    config.data.target_max_samples = 5000


    config.training.collision_penalty_wt = 0.0001   # same as learning rate scaling for the interpolant
    config.training.spatial_sigma = 10.0
    config.training.temporal_sigma = 0.1
    config.training.num_workers = 4
    config.training.use_lpips_spatial = False

    return config