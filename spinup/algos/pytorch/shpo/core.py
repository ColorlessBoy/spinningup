import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Tool Functions ===============================================
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def cg(self, A, b, iters=10, accuracy=1e-10):
    start_time = time()
    # A is a function: x ==> A(s) = A @ x
    x = torch.zeros_like(b)
    d = b.clone()
    g = -b.clone()
    g_dot_g_old = torch.tensor(1.)
    for _ in range(iters):
        g_dot_g = torch.dot(g, g)
        d = -g + g_dot_g / g_dot_g_old * d
        Ad = A(d)
        alpha = g_dot_g / torch.dot(d, Ad)
        x += alpha * d
        if g_dot_g < accuracy:
            break
        g_dot_g_old = g_dot_g
        g += alpha * Ad
    print("The cg() uses {}s.".format(time() - start_time))
    return x

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
# ===== End Of Tool Functions ===============================================

# ===== All About Networks ===========================================
def _weight_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight, gain=0.01)
        module.bias.data.zero_()

def mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)

class GenerativeGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.epsilon_dim = act_dim * act_dim
        hidden_sizes[0] += self.epsilon_dim
        self.net = mlp([obs_dim+self.epsilon_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh())
        self.act_limit = act_limit
        self.apply(_weight_init)

    def forward(self, obs, std=1.0, noise='gaussian', epsilon_limit=5.0):
        if noise == 'gaussian':
            # N[0, 1]^d
            epsilon = (std * torch.randn(obs.shape[0], self.epsilon_dim, device=obs.device)).clamp(-epsilon_limit, epsilon_limit)
        else:
            # U[-1, 1]^d
            epsilon = torch.rand(obs.shape[0], self.epsilon_dim, device=obs.device) * 2 - 1
        pi_action = self.net(torch.cat([obs, epsilon], dim=-1))
        pi_action = self.act_limit * pi_action
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

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.LeakyReLU(negative_slope=0.2)):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = GenerativeGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, self.act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

        self.obs_mean = torch.FloatTensor([0.0])
        self.obs_std = torch.FloatTensor([0.0])

    def act(self, obs, deterministic=False, noise='gaussian', obs_limit=5.0):
        obs = ((obs - self.obs_mean.to(obs.device))/(self.obs_std.to(obs.device) + 1e-8)).clamp(-obs_limit, obs_limit)
        with torch.no_grad():
            if deterministic:
                # For test, noise is N(0, 0.5)
                a = self.pi(obs, std=0.5, noise=noise)
            else:
                # For training, noise is N(0, 1.0)
                a = self.pi(obs, noise=noise)
        return a.detach().cpu().numpy()[0]
# ===== End Of Networks ====================================================

# ===== Replay Buffer ======================================================
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size=1000000, obs_limit=5.0):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        # For state normalization.
        self.total_num = 0

        self.obs_limit = 5.0
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_square_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std = np.zeros(obs_dim, dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

        self.total_num += 1
        self.obs_mean = self.obs_mean / self.total_num * (self.total_num - 1) + np.array(obs) / self.total_num
        self.obs_square_mean = self.obs_square_mean / self.total_num * (self.total_num - 1) + np.array(obs)**2 / self.total_num
        self.obs_std = np.sqrt(self.obs_square_mean - self.obs_mean ** 2)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_encoder(self.obs_buf[idxs]),
                     obs2=self.obs_encoder(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def sample_recently(self, batch_size=32):
        # Return recent samples / On-policy.
        if self.ptr >= batch_size:
            idxs = np.arange(self.ptr - batch_size, self.ptr)
        else:
            idxs = np.hstack(np.arange(self.ptr), np.arange(max_size-batch_size+self.ptr, max_size))
        batch = dict(obs=self.obs_encoder(self.obs_buf[idxs]),
                     obs2=self.obs_encoder(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def obs_encoder(self, o):
        return ((np.array(o) - self.obs_mean)/(self.obs_std + 1e-8)).clip(-self.obs_limit, self.obs_limit)
# ===== Enf Of Replay Buffer =============================================================================