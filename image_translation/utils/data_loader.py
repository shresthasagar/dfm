"""
Data loading utilities for image translation tasks.
Includes datasets for labeled images and VAE embeddings.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
from typing_extensions import Literal
import math
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.data import ImageFolder

class LabeledDataset(Dataset):
    """Dataset class for labeled images with attribute conditioning."""

    def __init__(self,
                 input_folder: str = 'data/celebahq2bitmoji',
                 domain: str = 'A',
                 keys: Optional[List[str]] = None,
                 train: bool = True,
                 rotate: bool = False,
                 crop: bool = False,
                 crop_size: float = 1.0,
                 new_size: int = 256,
                 horizontal_flip: bool = False,
                 low_pass: bool = False,
                 max_samples: Optional[int] = None):
        """
        Args:
            input_folder: Path to dataset folder
            domain: Domain A or B
            keys: List of attribute keys to condition on
            train: Whether this is training set
            rotate: Apply random rotation
            crop: Apply center crop
            crop_size: Size of crop as fraction of image
            new_size: Size to resize images to
            horizontal_flip: Apply random horizontal flip
            low_pass: Apply low pass filter
            max_samples: Maximum number of samples to use
        """
        # Validate inputs
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
        if keys is None:
            keys = []

        # Set up data paths
        split_suffix = 'train' if train else 'test'
        if domain == 'A':
            folder_path = os.path.join(input_folder, f'{split_suffix}A')
            attribute_path = os.path.join(input_folder, f'{split_suffix}A_attr.csv')
        else:
            folder_path = os.path.join(input_folder, f'{split_suffix}B')
            attribute_path = os.path.join(input_folder, f'{split_suffix}B_attr.csv')

        # Build image transformations
        base_transforms = [
            transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        # Apply low-pass filter if requested
        if low_pass:
            base_transforms.insert(0, transforms.Resize((new_size // 2, new_size // 2),
                               interpolation=transforms.InterpolationMode.BICUBIC))

        # Build optional transformations
        all_transforms = []
        if crop:
            all_transforms.extend([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((int(256 * crop_size), int(256 * crop_size)))
            ])

        if train and horizontal_flip:
            all_transforms.append(transforms.RandomHorizontalFlip(p=0.5))

        if rotate:
            all_transforms.append(transforms.RandomRotation((-90, -90), fill=255))

        all_transforms.extend(base_transforms)
        self.transform = transforms.Compose(all_transforms)

        # Load and process data
        image_metadata = self._load_attributes(attribute_path, folder_path)
        self.image_paths = image_metadata['image_id'].values

        # Select only requested attribute columns
        available_keys = [k for k in keys if k in image_metadata.columns]
        if available_keys:
            self.attribute_labels = image_metadata[available_keys].astype(int)
        else:
            # No valid keys found, create dummy labels
            self.attribute_labels = pd.DataFrame({'dummy': [1] * len(image_metadata)}, index=image_metadata.index)

        print(f"Loaded dataset with {len(self.image_paths)} images and {len(self.attribute_labels.columns)} attributes")

        # Limit samples if requested
        self.max_samples = max_samples if max_samples is not None else len(self.image_paths)

    def _load_attributes(self, attr_path: str, image_dir: str) -> pd.DataFrame:
        """Load attribute file and create image paths."""
        assert os.path.exists(attr_path), f'Attribute file {attr_path} does not exist'
        df = pd.read_csv(attr_path)

        # Create full image paths
        df['image_id'] = df['image_id'].apply(lambda x: os.path.join(image_dir, x))
        return df

    def __len__(self) -> int:
        """Returns dataset length."""
        return min(self.max_samples, len(self.image_paths))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns image and label pair."""
        image_path = self.image_paths[idx]
        image_array = plt.imread(image_path).copy()

        # Handle grayscale images
        if len(image_array.shape) == 2:
            image_array = np.repeat(np.expand_dims(image_array, axis=2), 3, axis=2)

        # Convert float images to uint8 if needed
        if image_array.dtype == 'float32' and image_array.max() <= 1.0 and image_array.min() >= 0.0:
            image_array = (image_array * 255).astype(np.uint8)

        # Apply transformations
        transformed_image = self.transform(Image.fromarray(image_array))

        # Get labels
        labels = torch.tensor(self.attribute_labels.iloc[idx].values, dtype=torch.float32)

        return transformed_image, labels


def save_vae_embeddings(config, train_loader1, train_loader2, vae, workdir, horizontal_flip=True):
    """Pre-computes and saves VAE embeddings for both domains."""
    embedding_dir = os.path.join(workdir, config.data.vae_name, 'vae_embeddings')
    
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir, exist_ok=True)
    elif (os.path.exists(os.path.join(embedding_dir, 'source_embeddings.pt')) and 
          os.path.exists(os.path.join(embedding_dir, 'target_embeddings.pt'))):
        print("VAE embeddings exist. Using existing files.")
        return

    for domain, loader in [('source', train_loader1), ('target', train_loader2)]:
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Encoding {domain} domain"):
                images, batch_labels = batch
                images = images.cuda()
                
                # Get embeddings for original images
                encoded = vae.encode(images).latent_dist.mode().cpu()
                embeddings.append(encoded)
                labels.append(batch_labels)
                
                # Get embeddings for flipped images if enabled
                if horizontal_flip:
                    images_flipped = torch.flip(images, dims=[3])
                    encoded_flipped = vae.encode(images_flipped).latent_dist.mode().cpu()
                    embeddings.append(encoded_flipped)
                    labels.append(batch_labels)
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        
        torch.save({'embeddings': embeddings, 'labels': labels},
                  os.path.join(embedding_dir, f'{domain}_embeddings.pt'))

    print("VAE embeddings saved successfully.")


def get_vae_embedding_loader(config, workdir, domain, train=True, 
                           aligned_conditional_samples=False, max_samples=None,
                           horizontal_flip=True):
    """Creates dataloader for VAE embeddings."""
    embedding_path = f"{workdir}/{config.data.vae_name}/vae_embeddings/{domain}_embeddings.pt"
    dataset = VAEEmbeddingDataset(embedding_path, aligned_conditional_samples=aligned_conditional_samples,
                                 max_samples=max_samples, horizontal_flip=horizontal_flip)
    print(f"Dataset length: {len(dataset)}")
    
    batch_size = (config.training.batch_size//config.data.num_conditionals 
                  if aligned_conditional_samples else config.training.batch_size)
    
    return DataLoader(dataset=dataset,
                     batch_size=batch_size,
                     shuffle=train,
                     num_workers=config.training.num_workers,
                     pin_memory=True,
                     drop_last=True)


class VAEEmbeddingDataset(Dataset):
    """Dataset for pre-computed VAE embeddings."""
    
    def __init__(self, embedding_path, aligned_conditional_samples=False, 
                 max_samples=None, horizontal_flip=True):
        """
        Args:
            embedding_path: Path to saved embeddings
            aligned_conditional_samples: Return aligned samples for each condition
            max_samples: Maximum number of samples
            horizontal_flip: Whether horizontal flips were used
        """
        data = torch.load(embedding_path)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        self.aligned_conditional_samples = aligned_conditional_samples

        if self.aligned_conditional_samples:
            self.label_indices = {}
            for i in range(self.labels.shape[1]):
                self.label_indices[i] = torch.where(self.labels[:, i] == 1)[0]
                
        dataset_multiplier = 2 if horizontal_flip else 1
        if max_samples is not None:
            if len(self.embeddings) > max_samples:
                print(f"Using max samples: {max_samples}")
                num_conditions = len(self.label_indices)
                self.label_indices = {i: self.label_indices[i][:(max_samples//num_conditions)*dataset_multiplier] 
                                    for i in self.label_indices}
        else:
            print("Using all samples")

    def __len__(self):
        if self.aligned_conditional_samples:
            return min(len(indices) for indices in self.label_indices.values())
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.aligned_conditional_samples:
            embeddings = []
            for label, indices in self.label_indices.items():
                randidx = indices[torch.randint(0, len(indices), (1,)).item()]
                embeddings.append(self.embeddings[randidx])
            return embeddings, torch.arange(len(embeddings))
        return self.embeddings[idx], self.labels[idx]



def get_loader(config: Any,
               domain: str,
               train: bool = True,
               crop: bool = False,
               crop_size: float = 1.0,
               max_samples: Optional[int] = None) -> DataLoader:
    """Creates dataloader for the specified domain.

    Args:
        config: Configuration object containing data and training parameters
        domain: Dataset domain type ('bitmoji', 'celebaForBitmoji', etc.)
        train: Whether this is for training
        crop: Whether to apply center cropping
        crop_size: Size of crop as fraction of image
        max_samples: Maximum number of samples to use

    Returns:
        DataLoader configured for the specified domain
    """
    # Set domain-specific parameters
    if domain == 'bitmoji':
        target_domain = 'B'
        attribute_keys = getattr(config.data, 'attributes', ['Male', '~Male'])
    elif domain == 'celebaForBitmoji':
        target_domain = 'A'
        attribute_keys = getattr(config.data, 'attributes', ['Male', '~Male'])
    else:
        raise NotImplementedError(f'Domain {domain} not implemented')

    # Create dataset
    dataset = LabeledDataset(
        input_folder=config.data.path,
        keys=attribute_keys,
        domain=target_domain,
        crop=crop,
        crop_size=crop_size,
        new_size=config.data.shape[2],
        train=train,
        horizontal_flip=True,  # Enable horizontal flip for training
        max_samples=max_samples
    )

    return DataLoader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=train,
        num_workers=config.training.num_workers,
        drop_last=True
    )


def create_data_loaders(config: Any, workdir: str) -> Dict[str, Any]:
    """Creates all necessary data loaders for training and evaluation.

    Args:
        config: Configuration object containing data and training parameters
        workdir: Working directory path

    Returns:
        Dictionary containing all data loaders and related objects
    """
    import logging

    logging.info('Preparing dataset...')

    # Create training data loaders
    train_loader_src = get_loader(
        config,
        domain=config.data.source,
        train=True,
        max_samples=config.data.source_max_samples,
        crop=config.data.source_center_crop,
        crop_size=config.data.source_crop_ratio
    )

    train_loader_tgt = get_loader(
        config,
        domain=config.data.target,
        train=True,
        max_samples=config.data.target_max_samples,
        crop=config.data.target_center_crop,
        crop_size=config.data.target_crop_ratio
    )

    # Create test data loaders for visualization
    test_loader_src = get_loader(
        config,
        domain=config.data.source,
        train=False,
        crop=config.data.source_center_crop,
        crop_size=config.data.source_crop_ratio
    )

    # Create test data loaders for FID calculation
    test_fid_loader_src = get_loader(
        config,
        domain=config.data.source,
        train=True,
        crop=config.data.source_center_crop,
        crop_size=config.data.source_crop_ratio
    )

    test_fid_loader_tgt = get_loader(
        config,
        domain=config.data.target,
        train=True,
        crop=config.data.target_center_crop,
        crop_size=config.data.target_crop_ratio
    )

    # Prepare display images for visualization
    test_display_images_src = torch.stack([
        test_loader_src.dataset[i][0] for i in range(min(16, len(test_loader_src.dataset)))
    ]).cuda()

    return {
        'train_loader_src': train_loader_src,
        'train_loader_tgt': train_loader_tgt,
        'test_loader_src': test_loader_src,
        'test_fid_loader_src': test_fid_loader_src,
        'test_fid_loader_tgt': test_fid_loader_tgt,
        'test_display_images_src': test_display_images_src,
    }


def prepare_fid_calculator(config: Any, test_fid_loader_tgt: DataLoader) -> Optional[Any]:
    """Prepares FID calculator if evaluation is enabled.

    Args:
        config: Configuration object
        test_fid_loader_tgt: Target domain FID loader

    Returns:
        FID calculator instance or None if evaluation is disabled
    """
    if config.training.eval_freq > 0:
        print('Preparing FID statistics for the real images...')
        with torch.no_grad():
            # Use min to avoid index errors if dataset is smaller than 1000
            num_samples = min(1000, len(test_fid_loader_tgt.dataset))
            target_images_real = torch.stack([
                test_fid_loader_tgt.dataset[i][0] for i in range(num_samples)
            ])

            # Import here to avoid circular imports
            from utils.fid import FIDImages
            fid_calculator = FIDImages(device='cuda', images_real=target_images_real)

        del target_images_real
        print('FID statistics for the real images prepared')
        return fid_calculator

    return None