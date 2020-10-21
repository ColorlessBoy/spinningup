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

def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def getSize(L, kernel, stride):
    return (L - kernel) // stride + 1

class MLPStateFeature(nn.Module):
    def __init__(self, obs_dim, feature_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(obs_dim[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        hw = getSize(np.array(obs_dim[1:]), 8, 4)
        hw = getSize(hw, 4, 2)
        hw = getSize(hw, 3, 1)

        self.fc1   = nn.Linear(hw[0] * hw[1] * 64, feature_dim)
        self.leakyrelu = [nn.LeakyReLU(0.1) for _ in range(4)]
        
        self.apply(_weight_init)
    
    def forward(self, s):
        s = self.leakyrelu[0](self.conv1(s))
        s = self.leakyrelu[1](self.conv2(s))
        s = self.leakyrelu[2](self.conv3(s))
        s = s.view(s.shape[0], -1)
        s = self.leakyrelu[3](self.fc1(s))
        return s

class GenerativeGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, feature_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.epsilon_dim = act_dim * act_dim
        hidden_sizes[0] += self.epsilon_dim
        self.net = mlp([feature_dim+self.epsilon_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh())
        self.feature = MLPStateFeature(obs_dim, feature_dim)
        self.apply(_weight_init)

    def forward(self, obs, std=1.0, noise='gaussian', epsilon_limit=5.0):
        obs = self.feature(obs)
        if noise == 'gaussian':
            epsilon = (std * torch.randn(obs.shape[0], self.epsilon_dim, device=obs.device)).clamp(-epsilon_limit, epsilon_limit)
        else:
            epsilon = torch.rand(obs.shape[0], self.epsilon_dim, device=obs.device) * 2 - 1
        pi_action = self.net(torch.cat([obs, epsilon], dim=-1))
        return pi_action

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, feature_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        hidden_sizes[0] += act_dim
        self.q = mlp([feature_dim+ act_dim] + list(hidden_sizes) + [1], activation)
        self.feature = MLPStateFeature(obs_dim, feature_dim)
        self.apply(_weight_init)

    def forward(self, obs, act):
        obs = self.feature(obs)
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, feature_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()

        # build policy and value functions
        self.pi = GenerativeGaussianMLPActor(obs_dim, feature_dim, act_dim, hidden_sizes, activation)
        self.q1 = MLPQFunction(obs_dim, feature_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, feature_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False, noise='gaussian', obs_limit=5.0):
        with torch.no_grad():
            if deterministic:
                a = self.pi(obs, std=0.5, noise=noise)
            else:
                a = self.pi(obs, noise=noise)
        return a.detach().cpu().numpy()[0]

# ========== Env Wrapper =======================================================
class EnvWrapper:
    def __init__(self, env):
        self.env = env
        obs = self.reset()
        self.obs_dim = obs.shape
        self.act_dim = env.action_space.n
        self.act_limit = 1.0
    
    def reset(self):
        obs = self.env.reset()
        obs = obs.transpose((2, 0, 1)) / 127.5 - 1.0
        return obs
    
    def step(self, action):
        action = np.argmax(action)
        obs, r, d, info = self.env.step(action)
        obs = obs.transpose((2, 0, 1)) / 127.5 - 1.0
        return obs, r, d, info
    
    def sample(self):
        return (2 * np.random.rand(self.act_dim) - 1) * self.act_limit
    
    def seed(self, s):
        self.env.seed(s)

    def render(self):
        self.env.render()
# ========== Env Wrapper =======================================================

# Maximum Mean Discrepancy
# geomloss: https://github.com/jeanfeydy/geomloss

class Sqrt0(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    return Sqrt0.apply(x)

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

    if kernel in kernel_routines:
        kernel = kernel_routines[kernel]

    K_xx = kernel(x, x).mean()
    K_xy = kernel(x, y).mean()
    K_yy = kernel(y, y).mean()

    return sqrt_0(K_xx + K_yy - 2*K_xy)

if __name__ == '__main__':
    max_z = 0
    avg_z = 0
    min_z = 100
    batch = 1000
    for _ in range(1000):
        # x = torch.randn(batch, 2)
        x = torch.rand(batch, 2) * 2 - 1
        y = torch.rand(batch, 2) * 2 - 1
        z = mmd(x, y, kernel='gaussian')
        avg_z += z
        max_z = max(max_z, z)
        min_z = min(min_z, z)
    print(max_z)
    print(min_z)
    print(avg_z/1000.0)