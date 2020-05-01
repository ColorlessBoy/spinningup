import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def _weight_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Generator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(obs_dim+act_dim, 256, normalize=False),
            *block(256, 256),
            *block(256, 256),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.apply(_weight_init)

    def forward(self, obs, std=1.0):
        epsilon = std * torch.randn(obs.shape[0], self.act_dim, device=obs.device)
        pi_action = self.model(torch.cat([obs, epsilon], dim=-1))
        pi_action = self.act_limit * pi_action
        return pi_action

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.apply(_weight_init)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(obs_dim+act_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, obs, act):
        q = self.model(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.LeakyReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = Generator(obs_dim, act_dim, hidden_sizes, activation, self.act_limit)
        self.q1 = Discriminator(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = Discriminator(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            if deterministic:
                a = self.pi(obs, std=0.5)
            else:
                a = self.pi(obs)
        return a.detach().cpu().numpy()[0]

# Maximum Mean Discrepancy
# geomloss: https://github.com/jeanfeydy/geomloss

def squared_distances(x, y):
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    elif x.dim() == 3:  # Batch computation
        D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2*D_xy + D_yy

def gaussian_kernel(x, y, blur=1.0):
    C2 = squared_distances(x / blur, y / blur)
    return (- .5 * C2 ).exp()

def energy_kernel(x, y, blur=None):
    return -squared_distances(x, y)

kernel_routines = {
    "gaussian" : gaussian_kernel,
    "energy"   : energy_kernel,
}

def mmd(x, y, kernel='gaussian'):
    b = x.shape[0]
    m = x.shape[1]
    n = y.shape[1]

    if kernel == 'energy':
        tr_xx = torch.tensor(0.0)
        tr_yy = torch.tensor(0.0)
    else:
        tr_xx = torch.tensor(1.0)
        tr_yy = torch.tensor(1.0)

    if kernel in kernel_routines:
        kernel = kernel_routines[kernel]

    K_xx = (kernel(x, x).mean()*m - tr_xx) / (m - 1)
    K_xy = kernel(x, y).mean()
    K_yy = (kernel(y, y).mean()*n - tr_yy) / (n - 1)

    return K_xx + K_yy - 2*K_xy

