import ml_collections
import os

def get_celebahq2bimoji_config(config):
    config.task = "translation" 
    config.training.gamma = "constant"
    config.training.flow_sigma = 0.01
    # data
    config.data = data = ml_collections.ConfigDict()
    data.source = "celebaForBitmoji"
    data.target = "bitmoji"
    data.shape = [3, 256, 256]
    data.shuffle_buffer = 10_000
    data.source_center_crop = False
    data.source_crop_ratio = 1.0
    data.target_center_crop = False
    data.target_crop_ratio = 1.0
    data.max_samples = 5000
    data.num_conditionals = 4
    data.horizontal_flip = False
    data.imbalance = False

    # data path Needs to be specified
    data.path = 'data/celebahq2bitmoji'

    # training
    config.training.resume_step = 30000
    config.training.num_steps = 100000
    config.training.eval_freq = 10000
    config.training.print_freq = 1000
    config.training.use_lpips_spatial = False
    config.training.image_space_lpips = False
    config.training.eval_only = False
    
    # data
    config.data.map_forward = True
    config.data.precomputed_stats_file = "celebahq2bitmoji"
    config.data.eval_labels = [0, 1, 2, 3]

    config.eval.labelwise = False

    return config


