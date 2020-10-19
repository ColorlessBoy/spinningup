import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import time

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

class GenerativeGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.epsilon_dim = act_dim * act_dim
        hidden_sizes[0] += self.epsilon_dim
        self.net = mlp([obs_dim+self.epsilon_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh())
        self.apply(_weight_init)

    def forward(self, obs, std=1.0, noise='gaussian', epsilon_limit=5.0):
        if noise == 'gaussian':
            epsilon = (std * torch.randn(obs.shape[0], self.epsilon_dim, device=obs.device)).clamp(-epsilon_limit, epsilon_limit)
        else:
            epsilon = torch.rand(obs.shape[0], self.epsilon_dim, device=obs.device) * 2 - 1
        pi_action = self.net(torch.cat([obs, epsilon], dim=-1))
        return pi_action

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        hidden_sizes[0] += act_dim
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.apply(_weight_init)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()

        # build policy and value functions
        self.pi = GenerativeGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

        self.obs_mean = torch.FloatTensor([0.0])
        self.obs_std = torch.FloatTensor([0.0])

    def act(self, obs, deterministic=False, noise='gaussian', obs_limit=5.0):
        obs = ((obs - self.obs_mean.to(obs.device))/(self.obs_std.to(obs.device) + 1e-8)).clamp(-obs_limit, obs_limit)
        with torch.no_grad():
            if deterministic:
                a = self.pi(obs, std=0.5, noise=noise)
            else:
                a = self.pi(obs, noise=noise)
        return a.detach().cpu().numpy()[0]

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, obs_limit=5.0):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        # For state normalization.
        self.total_num = 0
        self.obs_limit = 5.0
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_square_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std = np.zeros(obs_dim, dtype=np.float32)
        self.obs_normalization = True

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

        if self.obs_normalization:
            self.total_num += 1
            self.obs_mean = self.obs_mean / self.total_num * (self.total_num - 1) + obs / self.total_num
            self.obs_square_mean = self.obs_square_mean / self.total_num * (self.total_num - 1) + obs**2 / self.total_num
            self.obs_std = np.sqrt(self.obs_square_mean - self.obs_mean ** 2 + 1e-8)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_encoder(self.obs_buf[idxs]),
                     obs2=self.obs_encoder(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def obs_encoder(self, o):
        return ((np.array(o) - self.obs_mean)/(self.obs_std + 1e-8)).clip(-self.obs_limit, self.obs_limit)

class PlanEnv:
    def __init__(self, env, controller_model, device, each_control_steps=1):
        self.env = env
        self.controller = torch.load(controller_model).to(device)
        self.each_control_steps = each_control_steps
        self.device = device

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = 2
        self.act_limit = 2.0
        self.control_act_limit = env.action_space.high[0]

        self.obs = None
        
        # set by hand.
        self.goal_offset = 3
        self.goal_dim = 16
    
    def reset(self):
        self.obs = self.env.reset()
        return self.obs
    
    def step(self, action, render=False, sleep_time=1e-3):
        r, d, info = 0, 0, {}
        info['cost'] = 0.0
        info['goal_met'] = 0.0
        for _ in range(self.each_control_steps):
            if render:
                self.env.render()
                time.sleep(sleep_time)

            self.obs, control_r, control_d, control_info = \
                self.env.step(self._get_control_action(action))
            r += control_r
            info['cost'] += control_info.get('cost', 0.0)
            info['goal_met'] += control_info.get('goal_met', 0.0)
            if control_d: break
        return self.obs, r, control_d, info
    
    def sample(self):
        return (2 * np.random.rand(self.act_dim) - 1) * self.act_limit
    
    def seed(self, s):
        self.env.seed(s)

    def render(self):
        self.env.render()

    def _get_control_action(self, action):
        o = self.obs
        o[self.goal_offset:self.goal_offset+self.goal_dim] = \
            self.env.get_goal_feature(action)
        o = torch.FloatTensor(o.reshape(1, -1)).to(self.device)
        control_action = self.controller.act(o)
        return control_action * self.control_act_limit

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