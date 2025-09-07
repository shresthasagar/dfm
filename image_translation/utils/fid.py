"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""


import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
from tqdm import tqdm, trange
import os
from PIL import Image
from torchvision import transforms


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


class FIDImages:
    def __init__(self, device='cuda', images_real=None):
        self.device = device
        self.inception = nn.DataParallel(InceptionV3()).to(device)
        self.inception.eval()
        if images_real is not None:
            self.mu_real, self.cov_real = self.calculate_mu_cov(images_real)
        else:
            self.mu_real, self.cov_real = None, None
    
    def calculate_mu_cov(self, images, batch_size=8):
        actvs = []
        for i in trange(0, len(images), batch_size):
            x = images[i:i+batch_size]
            actv = self.inception(x.to(self.device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu = np.mean(actvs, axis=0)
        cov = np.cov(actvs, rowvar=False)
        return mu, cov

    def calculate_fid(self, images_fake=None, images_real=None, batch_size=8):
        assert images_fake is not None, 'images_fake is None'
        assert images_real is not None or (self.mu_real is not None and self.cov_real is not None), 'images_real is None'
        with torch.no_grad():
            if self.mu_real is None and self.cov_real is None:
                self.mu_real, self.cov_real = self.calculate_mu_cov(images_real, batch_size=batch_size)
            mu_fake, cov_fake = self.calculate_mu_cov(images_fake, batch_size=batch_size)
        return calc_fid(mu_fake, cov_fake, self.mu_real, self.cov_real)

    def test_fid_correctness(self, image_folder, batch_size=8):
        import os
        from PIL import Image
        from torchvision import transforms

        # Load and preprocess images
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        images = []
        print(f"Loading images from {image_folder}")
        for img_file in tqdm(image_files[:200]):
            img_path = os.path.join(image_folder, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)

        images = torch.stack(images)

        # Split images into two halves
        split_idx = len(images) // 2
        images_real = images[:split_idx]
        images_fake = images[split_idx:]

        # print image statistics mean variance
        print(f"Mean of images: {images.mean()}")
        print(f"Variance of images: {images.var()}")

        # Calculate FID
        fid_score = self.calculate_fid(images_fake, images_real, batch_size)
        
        print(f"FID score between two halves of the dataset: {fid_score}")
        return fid_score

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_calculator = FIDImages(device)
    
    image_folder = '/nfs/stak/users/shressag/hpc-share/Projects/datasets_i2i/afhq_v2/train/cat'
    fid_calculator.test_fid_correctness(image_folder)

