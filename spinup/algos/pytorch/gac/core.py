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
        self.obs_std = torch.FloatTensor([1.0])
        self.goal_mean = torch.FloatTensor([0.0])
        self.goal_std = torch.FloatTensor([1.0])

    def act(self, obs, deterministic=False, noise='gaussian', obs_limit=5.0, goal_limit=5.0):
        obs = ((obs - self.obs_mean.to(obs.device))/(self.obs_std.to(obs.device) + 1e-8)).clamp(-obs_limit, obs_limit)

        with torch.no_grad():
            if deterministic:
                a = self.pi(obs, std=0.5, noise=noise)
            else:
                a = self.pi(obs, noise=noise)
        return a.detach().cpu().numpy()[0]

# Replay Buffer

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, max_epochs=1000, max_steps=1000, goal_dim=1, obs_limit=5.0):
        self.max_epochs = max_epochs
        self.max_steps  = max_steps

        self.obs_buf  = np.zeros((self.max_epochs, self.max_steps, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((self.max_epochs, self.max_steps, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((self.max_epochs, self.max_steps, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros((self.max_epochs, self.max_steps), dtype=np.float32)
        self.cost_buf = np.zeros((self.max_epochs, self.max_steps), dtype=np.float32)
        self.done_buf = np.zeros((self.max_epochs, self.max_steps), dtype=np.float32)

        self.ag_buf    = np.zeros((self.max_epochs, self.max_steps, goal_dim), dtype=np.float32)
        self.tg_buf   = np.zeros((self.max_epochs, self.max_steps, goal_dim), dtype=np.float32)

        self.epoch_ptr, self.step_ptr, self.epoch = 0, 0, 0

        # For state normalization.
        self.total_num = 0
        self.obs_limit = 5.0
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_square_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std = np.ones(obs_dim, dtype=np.float32)
        self.obs_normalization = True


        self.g_limit = 5.0
        self.g_mean = np.zeros(goal_dim, dtype=np.float32)
        self.g_square_mean = np.zeros(goal_dim, dtype=np.float32)
        self.g_std = np.ones(goal_dim, dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done, cost=0, 
            achieved_goal=0, target_goal=0):
        self.obs_buf[self.epoch_ptr][self.step_ptr]  = obs
        self.obs2_buf[self.epoch_ptr][self.step_ptr] = next_obs
        self.act_buf[self.epoch_ptr][self.step_ptr]  = act
        self.rew_buf[self.epoch_ptr][self.step_ptr]  = rew
        self.cost_buf[self.epoch_ptr][self.step_ptr] = cost
        self.done_buf[self.epoch_ptr][self.step_ptr] = done

        self.ag_buf[self.epoch_ptr][self.step_ptr]   = achieved_goal
        self.tg_buf[self.epoch_ptr][self.step_ptr]   = target_goal

        self.step_ptr += 1
        if self.step_ptr >= self.max_steps:
            self.step_ptr = 0
            self.epoch_ptr = (self.epoch_ptr + 1) % self.max_epochs
            self.epoch = min(self.epoch + 1, self.max_epochs)
            print("ReplayBuffer.epoch = {}".format(self.epoch))

        if self.obs_normalization:
            self.total_num += 1

            self.obs_mean = self.obs_mean / self.total_num * (self.total_num - 1) + obs / self.total_num
            self.obs_square_mean = self.obs_square_mean / self.total_num * (self.total_num - 1) + obs**2 / self.total_num
            self.obs_std = np.sqrt(self.obs_square_mean - self.obs_mean ** 2 + 1e-8)

            self.g_mean = self.g_mean / self.total_num * (self.total_num - 1) + achieved_goal / self.total_num
            self.g_square_mean = self.g_square_mean / self.total_num * (self.total_num - 1) + achieved_goal**2 / self.total_num
            self.obs_std = np.sqrt(self.obs_square_mean - self.obs_mean ** 2 + 1e-8)

    def sample_batch(self, batch_size=256):
        epoch_idxs = np.random.randint(0, self.epoch, size=batch_size)
        step_idxs  = np.random.randint(0, self.max_steps-1, size=batch_size)

        tg  = self.tg_buf[epoch_idxs, step_idxs]
        ag  = tg - self.ag_buf[epoch_idxs, step_idxs]
        ag2 = tg - self.ag_buf[epoch_idxs, step_idxs+1]

        obs  = self.obs_encoder(self.obs_buf[epoch_idxs, step_idxs])
        obs2 = self.obs_encoder(self.obs2_buf[epoch_idxs, step_idxs])

        batch = dict(obs=np.concatenate((obs, self.g_encoder(ag)), axis=-1),
                     obs2=np.concatenate((obs2, self.g_encoder(ag2)), axis=-1),
                     act=self.act_buf[epoch_idxs, step_idxs],
                     rew=self.rew_buf[epoch_idxs, step_idxs],
                     done=self.done_buf[epoch_idxs, step_idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def hindsight_sample_batch(self, batch_size=256, future=0.75):
        epoch_idxs = np.random.randint(0, self.epoch, size=batch_size)
        step_idxs  = np.random.randint(0, self.max_steps-1, size=batch_size)

        her_indexes = np.where(np.random.uniform(size=batch_size) < future)
        future_offset = np.random.uniform(size=batch_size) * (self.max_steps - step_idxs)
        future_offset = future_offset.astype(int)
        future_t = (step_idxs + 1 + future_offset)[her_indexes]
        future_g = self.ag_buf[epoch_idxs[her_indexes], future_t]
        # future_g is new target_goal sampled from achieved_goal.

        tg  = self.tg_buf[epoch_idxs, step_idxs]
        tg[her_indexes] = future_g

        ag  = tg - self.ag_buf[epoch_idxs, step_idxs]
        ag2 = tg - self.ag_buf[epoch_idxs, step_idxs+1]

        obs  = self.obs_encoder(self.obs_buf[epoch_idxs, step_idxs])
        obs2 = self.obs_encoder(self.obs2_buf[epoch_idxs, step_idxs])

        batch = dict(obs=np.concatenate((obs, self.g_encoder(ag)), axis=-1),
                     obs2=np.concatenate((obs2, self.g_encoder(ag2)), axis=-1),
                     act=self.act_buf[epoch_idxs, step_idxs],
                     rew=get_reward(ag, ag2),
                     done=self.done_buf[epoch_idxs, step_idxs])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def get_reward(ag, ag2):
        d = np.linalg.norm(ag, axis=-1)
        d2 = np.linalg.norm(ag2, axis=-1)
        r = d2 - d # copy from safety_gym.
        r[d2 < 1.5] += 1.0 #goal_reward
        return r

    def obs_encoder(self, o):
        return ((np.array(o) - self.obs_mean)/(self.obs_std + 1e-8)).clip(-self.obs_limit, self.obs_limit)
    
    def g_encoder(self, g):
        return ((np.array(g) - self.g_mean)/(self.g_std + 1e-8)).clip(-self.g_limit, self.g_limit)


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