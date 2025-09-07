# Standard library imports
import os
import sys
import time
import pickle
import logging
import datetime
import argparse

# Third-party imports
import yaml
import torch
import wandb
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from absl import app, flags
import blobfile as bf
from ml_collections.config_flags import config_flags

# Local imports
from src.trainer import Trainer
from utils.data_loader import (
    create_data_loaders,
    prepare_fid_calculator
)

# Enable CUDA optimizations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Configure command line flags
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config")
flags.DEFINE_string("workdir", "runs", "Work directory.")
flags.DEFINE_enum("mode", "train", ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("devices", None, "CUDA devices to use.")
flags.DEFINE_enum("train_type", "shared", ["shared", "baseline"], "Parameter sharing type.")
flags.mark_flags_as_required(["config"])

# Configure PyTorch settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')

def train(config, workdir, mode="train"):
    """Main training function.

    Args:
        config: Configuration object containing model and training parameters
        workdir: Working directory for saving checkpoints and logs
        mode: Running mode ("train" or "eval"). When "eval", loads checkpoint and
              generates translations for all test images without training.
    """
    # Create all necessary data loaders
    data_loaders = create_data_loaders(config, workdir)


    # Extract data loaders with more intuitive names
    train_loader_src = data_loaders['train_loader_src']
    train_loader_tgt = data_loaders['train_loader_tgt']
    test_loader_src = data_loaders['test_loader_src']
    test_fid_loader_src = data_loaders['test_fid_loader_src']
    test_fid_loader_tgt = data_loaders['test_fid_loader_tgt']
    test_display_images_src = data_loaders['test_display_images_src']

    # Initialize trainer
    trainer = Trainer(config)

    # If using VAE, we need to update the VAE embeddings with the actual VAE model
    if config.model.use_vae:
        from utils.data_loader import save_vae_embeddings, get_vae_embedding_loader

        # Save VAE embeddings with the actual VAE model
        save_vae_embeddings(
            config,
            train_loader_src,
            train_loader_tgt,
            trainer.vae,
            workdir,
            horizontal_flip=True
        )

    # Set up checkpoint directory
    os.makedirs(os.path.join(workdir, config.name, 'checkpoints'), exist_ok=True)
    checkpoint_dir = f"{os.getcwd()}/{workdir}/{config.name}/checkpoints"
    iterations = 1

    # Load checkpoint if resuming training
    if config.training.resume_ckpt:
        if 'ckpt_iter' in config.training:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{config.training.ckpt_iter}.pt')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint-current.pt')
        if os.path.isfile(checkpoint_path):            
            print(f'Loading latest checkpoint {checkpoint_path}')
            iterations = trainer.load_checkpoint(checkpoint_path)
        else:
            print(f'No checkpoint found at {checkpoint_path}')
            sys.exit()

    # Prepare labels for conditional generation
    labels = torch.cat([
        torch.ones(config.training.batch_size//config.data.num_conditionals)*i 
        for i in range(config.data.num_conditionals)
    ], dim=0)

    # Evaluation only mode
    if mode == "eval":
        with torch.no_grad():
            eval_images = torch.stack([
                test_loader_src.dataset[i][0] for i in range(len(test_loader_src.dataset))
            ]).cuda()
            fake_domain_tgt = trainer.sample(trainer.v_model_ema, eval_images, labels=None)
            domain_src = eval_images.detach().cpu().numpy()
            fake_domain_tgt = fake_domain_tgt.clamp_(-1, 1).detach().cpu().numpy()

            # Save results
            image_dict = {
                'real_src': domain_src,
                'fake_tgt': fake_domain_tgt
            }
            save_path = f"{os.getcwd()}/eval/{config.name}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(image_dict, f)
            print(f'Saved evaluation images to {save_path}')
            sys.exit()

    # Prepare FID calculator
    fid_calculator = prepare_fid_calculator(config, test_fid_loader_tgt)

    # Handle VAE embeddings if using VAE (update with actual VAE model)
    if config.model.use_vae:
        from utils.data_loader import get_vae_embedding_loader

        # Replace original loaders with VAE embedding loaders
        train_loader_src = get_vae_embedding_loader(
            config,
            workdir,
            domain='source',
            train=True,
            aligned_conditional_samples=True,
            max_samples=config.data.source_max_samples if config.data.imbalance else None
        )
        train_loader_tgt = get_vae_embedding_loader(
            config,
            workdir,
            domain='target',
            train=True,
            aligned_conditional_samples=True,
            max_samples=config.data.target_max_samples if config.data.imbalance else None
        )
    
    # Initialize wandb run
    run_name = config.name if config.name != '' else datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    wandb.init(
        project='uot-fm',
        group=config.group_name if hasattr(config, 'group_name') else None,
        name=run_name,
        config=config.to_dict(),
        resume="allow" if config.wandb_resume_run else "never",
        id=config.wandb_run_id if config.wandb_run_id != "" else None,
    )

    # Log dataset images to wandb if dataset is small
    for domain_name, loader in [('src', train_loader_src), ('tgt', train_loader_tgt)]:
        if len(loader.dataset) <= 100:
            images = []
            for j in range(len(loader.dataset)):
                if config.model.use_vae:
                    with torch.no_grad():
                        images.append(
                            trainer.vae.decode(
                                torch.stack(loader.dataset[j][0]).cuda()
                            ).sample.cpu()
                        )
                else:
                    images.append(torch.stack(loader.dataset[j][0]))
            nrow = 4
            images = torch.cat(images, dim=0)
            images = torchvision.utils.make_grid(
                images,
                nrow=nrow,
                normalize=True,
                value_range=(-1, 1)
            )
            wandb.log({f'Train_loader_{domain_name}': wandb.Image(images)}, step=0)  

    # Learning rate scheduler
    def get_lr_multiplier_cosine(step, total_steps, start_step=0):
        """Cosine learning rate schedule."""
        if step < start_step:
            return 1.0
        progress = (step - start_step) / (total_steps - start_step)
        return 0.5 * (1 + np.cos(np.pi * progress))

    # Main training loop
    with tqdm(
        total=config.training.num_steps,
        desc="Progress",
        unit="it",
        initial=iterations
    ) as pbar:
        while iterations <= config.training.num_steps:
            for (embeddings_src, labels_src), (embeddings_tgt, _) in zip(train_loader_src, train_loader_tgt):
                # Prepare batch data
                embeddings_src = torch.cat(embeddings_src)
                embeddings_tgt = torch.cat(embeddings_tgt)
                labels = labels_src.permute(1, 0).flatten()

                # Move to GPU and slice to batch size
                embeddings_src = embeddings_src[:config.training.batch_size].cuda()
                embeddings_tgt = embeddings_tgt[:config.training.batch_size].cuda()
                labels = labels[:config.training.batch_size].cuda()

                if not config.conditional_coupling and not config.use_parametrized_interpolant:
                    labels = torch.zeros(labels.shape[0], 1).cuda()

                # Update learning rate if using cosine schedule
                if config.optim.schedule == "cosine" and iterations % 2000 == 0:
                    lr_multiplier = get_lr_multiplier_cosine(
                        iterations,
                        config.training.num_steps,
                        start_step=config.optim.schedule_start_step
                    )
                    trainer.update_learning_rate(lr_multiplier)

                # Training step
                trainer.step(embeddings_src, embeddings_tgt, labels, iterations)

                # Logging
                if (iterations+1) % config.training.print_freq == 0:
                    trainer.log_err_console(iterations)
                    trainer.log_err_wandb(iterations)
                    wandb.log(
                        {'learning_rate': trainer.get_current_learning_rate()},
                        step=iterations
                    )
                
                # Save sample images
                if (iterations+1) % config.training.sample_freq == 0:
                    merged_images = trainer.save_image_eval(test_display_images_src)
                    wandb.log({'src_to_tgt': wandb.Image(merged_images)}, step=iterations)

                # Save checkpoints
                if (iterations+1) % config.training.ckpt_freq == 0:
                    trainer.save_checkpoint(
                        os.path.join(checkpoint_dir, f'checkpoint-{iterations}.pt'),
                        iterations
                    )
                    trainer.save_checkpoint(
                        os.path.join(checkpoint_dir, 'checkpoint-current.pt'),
                        iterations
                    )
                    print('Saved model checkpoint')
                
                # Evaluation
                if ((iterations + 1) % config.training.eval_freq == 0 and config.training.eval_freq > 0) or mode == "eval":
                    if fid_calculator is not None:
                        num_batches = 1000 // config.training.batch_size if 1000 % config.training.batch_size == 0 else 1000 // config.training.batch_size + 1

                        with torch.no_grad():
                            # Calculate FID score
                            logging.info('Collecting images for FID calculation...')
                            images_src_real = torch.stack([
                                test_fid_loader_src.dataset[i][0] for i in range(1000)
                            ])

                            logging.info('Translating images for FID calculation...')
                            images_tgt_fake = torch.cat([
                                trainer.sample(
                                    trainer.v_model_ema,
                                    images_src_real[i * config.training.batch_size: (i + 1) * config.training.batch_size].cuda(),
                                    labels=None,
                                    device='cuda',
                                    solver='euler'
                                ).cpu()
                                for i in range(num_batches)
                            ], dim=0)

                            logging.info('Calculating FID...')
                            fid_score = fid_calculator.calculate_fid(images_fake=images_tgt_fake)
                            del images_src_real, images_tgt_fake

                            wandb.log({'FID': fid_score}, step=iterations)
                            logging.info('FID: {}'.format(fid_score))

                iterations += 1
                pbar.update(1)

                if iterations > config.training.num_steps or mode == "eval":
                    break

def main(argv):
    """Main entry point."""
    # Create working directory
    if not os.path.exists(FLAGS.workdir):
        FLAGS.workdir = 'runs'

    # Set up logging
    bf.makedirs(f"{FLAGS.workdir}/logs")
    logger = logging.getLogger()
    file_stream = open(f"{FLAGS.workdir}/logs/{FLAGS.config.name}.txt", "w")
    handler = logging.StreamHandler(file_stream)
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")

    # Configure CUDA devices
    if FLAGS.devices is not None:
        logging.info(f"Using CUDA devices {FLAGS.devices}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.devices

    # Set wandb API key
    FLAGS.config.wandb_key = os.environ.get("WANDB_API_KEY", FLAGS.config.wandb_key)

    # Run training or evaluation
    if FLAGS.mode == "train":
        train(FLAGS.config, FLAGS.workdir, mode="train")

    if FLAGS.mode == "eval":
        # For evaluation mode, ensure we load checkpoint and set eval-only behavior
        FLAGS.config.training.resume_ckpt = True
        train(FLAGS.config, FLAGS.workdir, mode="eval")

    file_stream.close()

if __name__ == "__main__":
    app.run(main)
