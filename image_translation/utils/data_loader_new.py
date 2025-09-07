"""
Data loading utilities for image translation tasks.
Includes datasets for labeled images and VAE embeddings.
"""

import torch
from torchvision import transforms
from typing import Union
from typing_extensions import Literal
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from utils.data import ImageFolder
from tqdm import tqdm

class LabeledDataset(Dataset):
    """Dataset class for labeled images with attribute conditioning."""
    
    def __init__(self, 
                 input_folder='image_translation/data/celebahq2bitmoji',
                 domain='A',
                 keys=None,
                 train=True,
                 rotate=False,
                 crop=False,
                 crop_size=1.0,
                 new_size=256,
                 horizontal_flip=False, 
                 discriminator_dataset=False,
                 include_negatives=False,
                 low_pass=False,
                 max_samples=None,
                 combined_keys=None):
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
            discriminator_dataset: Return balanced samples for discriminator
            include_negatives: Include negative attributes
            low_pass: Apply low pass filter
            max_samples: Maximum number of samples to use
            combined_keys: List of combined attribute keys
        """
        assert os.path.exists(input_folder), f'input_folder {input_folder} does not exist'
        
        # Handle special case of MNIST at 32x32
        self.zero_pad = False
        if 'MNIST' in input_folder and new_size==32:
            self.zero_pad = True

        # Set attribute keys
        self.keys = set([key for item in combined_keys for key in item])

        print(f"Using attribute keys: {self.keys}")

        # Set up data paths
        if domain=='A':
            folder_path = os.path.join(input_folder, 'trainA' if train else 'testA')
            attribute_path = os.path.join(input_folder, 'trainA_attr.csv' if train else 'testA_attr.csv')
        else:
            folder_path = os.path.join(input_folder, 'trainB' if train else 'testB')
            attribute_path = os.path.join(input_folder, 'trainB_attr.csv' if train else 'testB_attr.csv')

        # Build transforms
        if self.zero_pad:
            transform_list = [
                transforms.Pad([2,], fill=0, padding_mode='constant'),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        else:
            transform_list = [
                transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        if low_pass:
            transform_list = [transforms.Resize((new_size//2, new_size//2), 
                            interpolation=transforms.InterpolationMode.BICUBIC)] + transform_list

        assert 0 < crop_size <= 1, 'crop_size must be between 0 and 1'
        
        # Add optional transforms
        crop_transform = [transforms.Resize((256, 256)), 
                         transforms.CenterCrop((int(256*crop_size), int(256*crop_size)))] if crop else []
        rand_rotate = [transforms.RandomRotation((-90, -90), fill=255)] if rotate else []
        horizontal_flip = [transforms.RandomHorizontalFlip(p=0.5)] if (train and horizontal_flip) else []
        
        transform_list = crop_transform + horizontal_flip + rand_rotate + transform_list
        self.transform = transforms.Compose(transform_list)

        # Load image data and attributes
        image_data = self.read_attr_file(attribute_path, folder_path)
        self.files = image_data['image_id'].values
        self.labels = pd.DataFrame()

        # Process attribute labels
        for key in self.keys:
            if key not in image_data.columns:
                if key.startswith('~') and key.strip('~') in image_data.columns:
                    print(f"Using negation of {key.strip('~')} as {key}")
                    self.labels[key] = list(map(lambda x: 0 if (int(x)==1) else 1, 
                                              image_data[key.strip('~')]))
                else:
                    raise ValueError(f"Key {key} and its negation {key.strip('~')} not found")
            else:
                self.labels[key] = list(map(lambda x: max(int(x), 0), image_data[key]))
                if include_negatives:
                    self.labels['~'+key] = list(map(lambda x: 0 if x else 1, image_data[key]))

        print("Label statistics:")
        print(self.labels.head())
        
        # Group images by labels
        self.image_by_labels = []
        if combined_keys is None:
            for key in self.keys:
                self.image_by_labels.append([self.files[j] for j in range(len(self.files)) 
                                           if self.labels[key][j]==1])
                if include_negatives:
                    self.image_by_labels.append([self.files[j] for j in range(len(self.files))
                                               if self.labels['~'+key][j]==1])
        else:
            self.combined_labels = {tuple(key): [all(self.labels[k][j] == 1 for k in key) 
                                               for j in range(len(self.files))] 
                                  for key in combined_keys}
            self.labels = pd.DataFrame(self.combined_labels).astype(int)
            for key in combined_keys:
                self.image_by_labels.append([self.files[j] for j in range(len(self.files))
                                           if self.labels[tuple(key)][j] == 1])

        # Handle max samples limit
        self.discriminator_dataset = discriminator_dataset
        self.max_samples = max_samples if max_samples is not None else len(self.files)
        if max_samples is not None:
            if len(self.files) > max_samples:
                num_conditions = len(self.image_by_labels)
                self.image_by_labels = [self.image_by_labels[i][:max_samples//num_conditions] 
                                      for i in range(num_conditions)]
            else:
                print(f"Max samples {max_samples} exceeds dataset size {len(self.files)}. Using all.")

    def get_conditional_sizes(self):
        """Returns number of samples for each condition."""
        return [len(images) for images in self.image_by_labels]

    def read_attr_file(self, attr_path, image_dir):
        """Reads attribute file or creates dummy attributes if not found."""
        if os.path.exists(attr_path):
            with open(attr_path) as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            columns = lines[0].split(',')
            items = [line.split(',') for line in lines[1:]]
        else:
            print("Creating dummy attributes")
            columns = ['image_id', 'dummy']
            items = [[x, 1] for x in os.listdir(image_dir)]

        df = pd.DataFrame(items, columns=columns)
        df['image_id'] = df['image_id'].map(lambda x: os.path.join(image_dir, x))
        return df

    def __len__(self):
        """Returns dataset length based on mode."""
        if not self.discriminator_dataset:
            return len(self.files[:self.max_samples])
        return min(len(images) for images in self.image_by_labels)
    
    def __getitem__(self, idx):
        """Returns image and label pair."""
        if not self.discriminator_dataset:
            # Regular dataset mode
            image = plt.imread(self.files[idx]).copy()
            if len(image.shape) == 2:
                image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
            
            if image.dtype == 'float32' and image.max()<=1.0 and image.min()>=0.0:
                image = (image*255).astype(np.uint8)
                
            image = self.transform(Image.fromarray(image))
            label = torch.tensor(self.labels.iloc[idx])
            return image, label
        
        else:
            # Discriminator balanced sampling mode
            images = []
            for i in range(len(self.keys)):
                randidx = np.random.randint(0, len(self.image_by_labels[i]))
                image = plt.imread(self.image_by_labels[i][randidx]).copy()
                
                if len(image.shape) == 2:
                    image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)

                if image.dtype == 'float32' and image.max()<=1.0 and image.min()>=0.0:
                    image = (image*255).astype(np.uint8)
                    
                image = self.transform(Image.fromarray(image))
                images.append(image)
            return images, torch.arange(len(images))

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

class DefaultDataset(Dataset):
    """Basic dataset without attribute conditioning."""
    
    def __init__(self, input_folder='../../dataset_i2i/selfie2anmie', domain='A',
                 train=True, rotate=False, crop=True, new_size=128, 
                 max_rotation=0, horizontal_flip=False):
        # Set up data path
        if domain=='A':
            folder_path = os.path.join(input_folder, 'trainA' if train else 'testA')
        else:
            folder_path = os.path.join(input_folder, 'trainB' if train else 'testB')

        # Build transforms
        transform_list = [
            transforms.Resize((new_size, new_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        
        rand_rotate = [transforms.RandomRotation((-90-max_rotation, -90+max_rotation), fill=255)] if rotate else []
        horizontal_flip = [transforms.RandomHorizontalFlip(p=0.5)] if (train and horizontal_flip) else []
        transform_list = horizontal_flip + rand_rotate + transform_list
        
        transform = transforms.Compose(transform_list)
        
        self.image_set = ImageFolder(folder_path, transform=transform)
        self.labels = torch.ones(len(self.image_set))
            
    def __len__(self):
        return len(self.image_set)
    
    def __getitem__(self, index):
        return self.image_set[index], self.labels[index]

def get_loader(config,
               domain:Union[str, Literal['mnist', 'rotatedmnist', 'shoes_edges', 'shoes',
                                       'rotatedshoes', 'bitmoji', 'celebaForBitmoji',
                                       'shoes_edges_2_cond', 'rotatedshoes_2_cond']],
               train=True,
               crop=False,
               crop_size=1.0,
               discriminator_dataset=False,
               max_samples=None,
               combined_keys=None) -> DataLoader:
    """Creates appropriate dataloader based on domain."""
    
    if domain == 'bitmoji':
        keys = (['Male', '~Male', 'Black_Hair', '~Black_Hair'] 
                if not hasattr(config.data, 'attributes') else config.data.attributes)
        dataset = LabeledDataset(
            input_folder=config.data.path,
            keys=keys,
            domain='B',
            rotate=False,
            crop=crop,
            crop_size=crop_size,
            new_size=config.data.shape[2],
            train=train,
            discriminator_dataset=discriminator_dataset,
            horizontal_flip=False,
            include_negatives=False,
            max_samples=max_samples,
            combined_keys=combined_keys
        )
                                 
    elif domain == 'celebaForBitmoji':
        keys = (['Male', '~Male', 'Black_Hair', '~Black_Hair']
                if not hasattr(config.data, 'attributes') else config.data.attributes)
        dataset = LabeledDataset(
            input_folder=config.data.path,
            keys=keys, 
            domain='A',
            rotate=False,
            crop=crop,
            crop_size=crop_size,
            new_size=config.data.shape[2],
            train=train,
            discriminator_dataset=discriminator_dataset,
            horizontal_flip=False,
            include_negatives=False,
            max_samples=max_samples,
            combined_keys=combined_keys
        )
    else:
        raise NotImplementedError(f'Domain {domain} not implemented')

    batch_size = (math.ceil(config.training.batch_size/config.data.num_conditionals)
                 if discriminator_dataset else config.training.batch_size)
                 
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=config.training.num_workers,
        drop_last=True
    )
