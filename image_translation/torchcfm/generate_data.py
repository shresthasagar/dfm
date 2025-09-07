from torchcfm.models.models import MLP

import torch
from typing import *
from functools import partial
import math

from sklearn.datasets import make_moons, make_swiss_roll
import numpy as np


CENTERS = {
    '8_gaussians':  [
        (1, 0),
        (1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
        (0, 1),
        (-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
        (-1, 0),
        (-1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
        (0, -1),
        (1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
    ],
    '2_gaussians': [
        (1,  1),
        (1, -1),
    ],
    '2_gaussians_overlap': [
        (2,  0.5),
        (2, -0.5),
    ],
    '3_gaussians_overlap': [
        (2,  0.5),
        (2, -0.5),
        (2.5,0.0),
    ]
}
functions = {
    'mlp': partial(MLP, time_varying=False),
    'reflection_origin': lambda x : -x,
    'rotate_90': lambda x : torch.stack([-x[..., 1], x[..., 0]], dim=-1),
}

# Create data generator class
class SquaresToCurves:
    def __init__(self,
                 square1_center: Tuple[float, float] = (0.0, 0.0),
                 square2_center: Tuple[float, float] = (2.0, 2.0),
                 square_width: float = 1.0,
                 moon_center: Tuple[float, float] = (0.0, 0.0),
                 spiral_center: Tuple[float, float] = (2.0, 2.0),
                 noise: float = 0.1):
        
        self.square1_center = torch.tensor(square1_center)
        self.square2_center = torch.tensor(square2_center)
        self.square_width = square_width
        self.moon_center = np.array(moon_center)
        self.spiral_center = np.array(spiral_center)
        self.noise = noise

    def get_data(self, n_samples=1000):
        # Ensure even split between components
        n_per_component = n_samples // 2
        
        # Generate source squares
        square1 = (torch.rand(n_per_component, 2) - 0.5) * self.square_width + self.square1_center
        square2 = (torch.rand(n_per_component, 2) - 0.5) * self.square_width + self.square2_center
        square1 = square1.type(torch.float32)
        square2 = square2.type(torch.float32)
        
        # Generate target distributions
        moons, _ = make_moons(n_samples=n_per_component, noise=self.noise)
        moons = torch.tensor(moons + self.moon_center).type(torch.float32)
        
        # Generate spiral (using only x,y from swiss_roll)
        spiral, _ = make_swiss_roll(n_samples=n_per_component, noise=self.noise)
        spiral = torch.tensor(spiral[:, [0, 2]]/10.0 + self.spiral_center).type(torch.float32)
        
        # Combine domains
        domain1 = torch.cat([square1, square2])
        domain2 = torch.cat([moons, spiral])
        
        # Create indices
        ind1 = torch.cat([torch.zeros(n_per_component), torch.ones(n_per_component)])
        ind2 = torch.cat([torch.zeros(n_per_component), torch.ones(n_per_component)])
        
        # Random permutation
        perm1 = torch.randperm(n_samples)
        perm2 = torch.randperm(n_samples)
        
        domain1 = domain1[perm1]
        domain2 = domain2[perm2]
        ind1 = ind1[perm1]
        ind2 = ind2[perm2]
        
        return domain1, domain2, ind1, ind2


# Create data generator class
class SquaresToCurves:
    def __init__(self,
                 square1_center: Tuple[float, float] = (0.0, 0.0),
                 square2_center: Tuple[float, float] = (2.0, 2.0),
                 square_width: float = 1.0,
                 moon_center: Tuple[float, float] = (0.0, 0.0),
                 spiral_center: Tuple[float, float] = (2.0, 2.0),
                 noise: float = 0.1):
        
        self.square1_center = torch.tensor(square1_center)
        self.square2_center = torch.tensor(square2_center)
        self.square_width = square_width
        self.moon_center = np.array(moon_center)
        self.spiral_center = np.array(spiral_center)
        self.noise = noise

    def get_data(self, n_samples=1000):
        # Ensure even split between components
        n_per_component = n_samples // 2
        
        # Generate source squares
        square1 = (torch.rand(n_per_component, 2) - 0.5) * self.square_width + self.square1_center
        square2 = (torch.rand(n_per_component, 2) - 0.5) * self.square_width + self.square2_center
        square1 = square1.type(torch.float32)
        square2 = square2.type(torch.float32)
        
        # Generate target distributions
        moons, _ = make_moons(n_samples=n_per_component, noise=self.noise)
        moons = torch.tensor(moons + self.moon_center).type(torch.float32)
        
        # Generate spiral (using only x,y from swiss_roll)
        spiral, _ = make_swiss_roll(n_samples=n_per_component, noise=self.noise)
        spiral = torch.tensor(spiral[:, [0, 2]]/10.0 + self.spiral_center).type(torch.float32)
        
        # Combine domains
        domain1 = torch.cat([square1, square2])
        domain2 = torch.cat([moons, spiral])
        
        # Create indices
        ind1 = torch.cat([torch.zeros(n_per_component), torch.ones(n_per_component)])
        ind2 = torch.cat([torch.zeros(n_per_component), torch.ones(n_per_component)])
        
        # Random permutation
        perm1 = torch.randperm(n_samples)
        perm2 = torch.randperm(n_samples)
        
        domain1 = domain1[perm1]
        domain2 = domain2[perm2]
        ind1 = ind1[perm1]
        ind2 = ind2[perm2]
        
        return domain1, domain2, ind1, ind2


class GMMData:
    def __init__(self, 
                    dim: int = 2, 
                    num_components: int = 2,
                    scale: float = 1.0, 
                    var: float = 1.0,
                    cond: bool = True, 
                    center_type: Literal['random', '8_gaussians', '2_gaussians']= 'random',
                    flow_type: Literal['mlp', 'reflection_origin'] = 'mlp'):

        self.dim = dim
        self.num_components = num_components
        self.scale = scale
        self.cond = cond
        self.normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
                    )
        if center_type == 'random':
            self.centers = [(torch.rand(num_components,dim) - 0.5) * 2.0]
        else:
            assert center_type in CENTERS, f"Invalid centers {center_type}"
            self.centers = CENTERS[center_type]
            self.centers = [torch.tensor(center + tuple([0]*(dim-2))) for center in self.centers]
        self.centers = [center * scale for center in self.centers]
        
        assert flow_type in functions, f"Invalid flow {flow_type}"
        if flow_type == 'mlp':
            self.flow = functions[flow_type](dim)
        else:
            self.flow = functions[flow_type]

    def get_data(self, n_samples=1000):
        # fetch samples from the first domain
        ind = []
        for j in range(self.num_components):
            ind += [j]*(n_samples//self.num_components) 
        ind + [self.num_components-1]*(n_samples - len(ind))
        ind = torch.tensor(ind)

        # sample the noise
        noise = self.normal_distribution.sample((n_samples,))

        # offset data by the component centers
        domain1 = []
        for i in range(self.num_components):
            domain1.append(self.centers[i] + noise[i*(n_samples//self.num_components):(i+1)*(n_samples//self.num_components)])
        domain1 = torch.cat(domain1)
        
        with torch.no_grad():
            domain2 = self.flow(domain1)
            # permute the samples
            perm = torch.randperm(len(domain1))
            domain2 = domain2[perm]
            ind2 = ind[perm]

            perm = torch.randperm(len(domain1))
            domain1 = domain1[perm]
            ind1 = ind[perm]

        
        if not self.cond:
            return domain1, domain2
        else:
            return domain1, domain2, ind1, ind2

    

if __name__=='__main__':
    data = GMMData(dim=2, num_components=2, scale=3.0, var=1.0, cond=True, center_type='2_gaussians', flow_type='mlp')
    d1, d2, ind = data.get_data(n_samples=1000)
    print(f'd1: {d1.shape}, d2: {d2.shape}, ind: {len(ind)}')


