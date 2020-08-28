from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.gsac.core as core
from spinup.utils.logx import EpochLogger

from geomloss import SamplesLoss

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, obs_limit=5.0):
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

def gsac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak_q=0.995, polyak_pi=0.0, lr=1e-3, 
        batch_size=100, start_steps=10000, 
        update_after=1000, update_every=2, update_steps=2,
        num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, 
        device='cuda', expand_batch=100, 
        start_beta_pi=1.0, beta_pi_velocity=0.0, max_beta_pi=1.0,
        start_beta_q =0.0, beta_q_velocity =0.0, max_beta_q =0.0,
        start_bias_q =0.0, bias_q_velocity =0.0, max_bias_q =0.0, 
        reward_scale=1.0, kernel='energy', noise='gaussian',
        beta_sh = 1.0, eta=100, sh_p=2, blur_loss=10, blur_constraint=1, scaling=0.95, backend="tensorized"):
    """
    Generative Actor-Critic (GAC)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    device = torch.device(device)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env, test_env = env_fn(), env_fn()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    test_env.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    print("obs_dim = {}, act_dim = {}".format(obs_dim, act_dim))

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_targ_params = itertools.chain(ac_targ.q1.parameters(), ac_targ.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    sinkhorn_divergence_con = SamplesLoss(loss="sinkhorn", p=sh_p, blur=blur_constraint, backend=backend, scaling=scaling,
                                      debias=True)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data, beta_q, bias_q):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        o = torch.FloatTensor(o).to(device)
        a = torch.FloatTensor(a).to(device).requires_grad_(True)
        r = torch.FloatTensor(r).to(device)
        o2 = torch.FloatTensor(o2).to(device)
        d = torch.FloatTensor(d).to(device)

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Q Net Regularization.
        gradients = torch.autograd.grad(
            outputs=q1.sum()+q2.sum(),
            inputs=a,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1).mean()
        gradient_penalty = (gradient_norm - bias_q) ** 2

        if beta_q <= 0.0:
            gradient_penalty.detach_()

        # Bellman backup for Q functions

        with torch.no_grad():
            # Target actions come from *current* policy
            a2 = ac_targ.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2 + beta_q*gradient_penalty

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy(),
                      Q_penalty=gradient_norm.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data, beta_pi, beta_sh):
        o = data['obs']
        o = torch.FloatTensor(o).to(device)

        o2 = o.repeat(expand_batch, 1)
        a2 = ac.pi(o2)
        q1_pi = ac.q1(o2, a2)
        q2_pi = ac.q2(o2, a2)
        q_pi = torch.min(q1_pi, q2_pi)

        a2 = a2.view(expand_batch, -1, a2.shape[-1]).transpose(0, 1).contiguous()
        with torch.no_grad():
            a3 = (2 * torch.rand_like(a2) - 1) * act_limit

        mmd_entropy = core.mmd(a2/act_limit, a3/act_limit, kernel=kernel)
        if beta_pi <= 0.0:
            mmd_entropy.detach_()

        a4 = ac_targ.pi(o2).view(expand_batch, -1, a2.shape[-1]).transpose(0, 1).contiguous()
        # weight = torch.ones(a4.shape[0], expand_batch, requires_grad=False, 
        #        dtype=torch.float32, device=device) / expand_batch
        # sh_penalty = sinkhorn_divergence_con(weight, a2/act_limit, weight, a4/act_limit)
        sh_penalty = core.mmd(a2/act_limit, a4/act_limit, kernel=kernel)
        if beta_sh <= 0.0:
            sh_penalty.detach_()

        # Entropy-regularized policy loss
        loss_pi = -q_pi.mean() + beta_pi * mmd_entropy + beta_sh * sh_penalty

        # Useful info for logging
        pi_info = dict(pi_penalty=mmd_entropy.detach().cpu().numpy(), 
                       sh_penalty=sh_penalty.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update_critic(data, beta_q, bias_q):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data, beta_q=beta_q, bias_q=bias_q)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)
    
    def update_targ_q(polyak_q):
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for q, q_targ in zip(q_params, q_targ_params):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                q_targ.data.mul_(polyak_q)
                q_targ.data.add_((1 - polyak_q) * q.data)

    def update_actor(data, beta_pi, beta_sh):
        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for q in q_params:
            q.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data, beta_pi=beta_pi, beta_sh=beta_sh)
        loss_pi.backward()
        pi_optimizer.step()

        for q in q_params:
            q.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

    def update_target_pi(polyak_pi):
        for p, p_targ in zip(ac.pi.parameters(), ac_targ.pi.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak_pi)
            p_targ.data.add_((1 - polyak_pi) * p.data)

    def get_action(o, deterministic=False):
        # o = replay_buffer.obs_encoder(o)
        o = torch.FloatTensor(o.reshape(1, -1)).to(device)
        a = ac_targ.act(o, deterministic, noise=noise)
        return a

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t <= start_steps:
            a = env.action_space.sample()
        else:
            a = get_action(o, deterministic=False)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)

        d = False if ep_len==max_ep_len else d

        # Reward Modified.
        if not d: r *= reward_scale

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)
        ac.obs_std = torch.FloatTensor(replay_buffer.obs_std).to(device)
        ac.obs_mean = torch.FloatTensor(replay_buffer.obs_mean).to(device)
        ac_targ.obs_std = ac.obs_std
        ac_targ.obs_mean = ac.obs_mean

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            epoch = (t+1) // steps_per_epoch
            beta_pi = min(start_beta_pi+beta_pi_velocity*epoch, max_beta_pi)
            beta_q = min(start_beta_q+beta_q_velocity*epoch, max_beta_q)
            bias_q = min(start_bias_q+bias_q_velocity*epoch, max_bias_q)
            
            # Firstly, update critic.
            for j in range(update_steps):
                batch = replay_buffer.sample_batch(batch_size)
                update_critic(data=batch, beta_q=beta_q, bias_q=bias_q)
                update_targ_q(polyak_q);
            
            # Secondly, update actor.
            # update_actor() uses ac.q1 and ac.q2.
            for j in range(update_steps):
                batch = replay_buffer.sample_batch(batch_size)
                update_actor(data=batch, beta_pi=beta_pi, beta_sh=beta_sh);
            update_target_pi(polyak_pi)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('pi_penalty', with_min_and_max=True)
            logger.log_tabular('sh_penalty', with_min_and_max=True)
            logger.log_tabular('Q_penalty', with_min_and_max=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # Normalize State
            print("obs_mean="+str(replay_buffer.obs_mean))
            print("obs_std=" +str(replay_buffer.obs_std))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='gac')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    gsac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

#   python -m spinup.run gac_pytorch --env HalfCheetah-v2 --exp_name sac_HalfCheetahv2