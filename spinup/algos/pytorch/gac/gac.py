from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import safety_gym
import time
import spinup.algos.pytorch.gac.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, obs_limit=5.0):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        # For state normalization.
        self.total_num = 0
        self.obs_limit = 5.0
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_square_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std = np.zeros(obs_dim, dtype=np.float32)

    def store(self, obs, act, rew, cost, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
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
                     cost=self.cost_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def obs_encoder(self, o):
        return ((np.array(o) - self.obs_mean)/(self.obs_std + 1e-8)).clip(-self.obs_limit, self.obs_limit)

def gac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, 
        replay_size=int(1e6), gamma=0.99, cost_gamma=0.99,
        polyak=0.995, pi_lr=1.0, lr=1e-3, 
        batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, 
        num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, 
        device='cuda', expand_batch=100, 
        alpha=-0.01, beta=2.0, 
        penalty=0.0, largest_penalty=1.0,
        cost_lr=1e-3, largest_cost=0.0, cost_scale=1.0, cost_bias=2.0,
        warm_steps=0, reward_scale=1.0, 
        kernel='energy', noise='gaussian',
        model_file=None):
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

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print("obs_dim = {}, act_dim = {}".format(obs_dim, act_dim))

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim, act_dim, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac)

    if model_file:
        model_parameters = torch.load(model_file)
        ac.load_state_dict(model_parameters['ac'])
        ac_targ.load_state_dict(model_parameters['ac'])

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters())
    cq_params = itertools.chain(ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer.obs_std = ac.obs_std.data.detach().cpu().numpy()
    replay_buffer.obs_mean = ac.obs_mean.data.detach().cpu().numpy()

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # auto-hyperparameters
    auto_alpha = False
    if alpha < 0.0:
        auto_alpha = True
        alpha = -alpha

    log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)

    # auto-penalty
    auto_penalty = False
    if penalty < 0.0:
        auto_penalty = True
        penalty = -penalty
    
    log_penalty = torch.tensor(np.log(penalty), requires_grad=True, device=device)

    # Set up function for computing Q-losses
    def compute_loss_q(data):
        o, a, r, c, o2, d = data['obs'], data['act'], data['rew'], data['cost'], data['obs2'], data['done']

        o = torch.FloatTensor(o).to(device)
        a = torch.FloatTensor(a).to(device)
        r = torch.FloatTensor(r).to(device)
        c = torch.FloatTensor(c).to(device)
        o2 = torch.FloatTensor(o2).to(device)
        d = torch.FloatTensor(d).to(device)

        q1 = ac.q1(o, a)

        q2 = ac.q2(o, a)

        # Bellman backup for Q functions

        with torch.no_grad():
            # Target actions come from *current* policy
            a2 = ac.pi(o2)
            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            backup = r + gamma * (1 - d) * q1_pi_targ

            # Cost Target Q-values
            q2_pi_targ = ac_targ.q2(o2, a2)
            cbackup = -c + cost_gamma * (1 - d) * q2_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q = loss_q1

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        # MSE loss against Bellman cbackup
        loss_q2 = ((q2 - cbackup)**2).mean()
        loss_cq = loss_q2

        return loss_q, loss_cq, q_info

    # Set up function for computing pi loss
    def compute_loss_pi(data):
        o = data['obs']
        o = torch.FloatTensor(o).to(device)

        o2 = o.repeat(expand_batch, 1)
        a2 = ac.pi(o2)
        q1_pi = ac.q1(o2, a2)
        q2_pi = ac.q2(o2, a2)

        a2 = a2.view(expand_batch, -1, a2.shape[-1]).transpose(0, 1)
        with torch.no_grad():
            a3 = (2 * torch.rand_like(a2) - 1)

        mmd_entropy = core.mmd(a2, a3, kernel=kernel)

        q1_pi_std = q1_pi.std().detach() + 1e-8
        q2_pi_std = q2_pi.std().detach() + 1e-8
        # Entropy-regularized policy loss
        loss_pi = -(q1_pi.mean()/q1_pi_std + penalty*q2_pi.mean()/q2_pi_std)/(1 + penalty) + alpha * mmd_entropy

        # Useful info for logging
        pi_info = dict(mmd_entropy=mmd_entropy.detach().cpu().numpy(),
                       Q1ValsStd = q1_pi_std,
                       Q2ValsStd = q2_pi_std)

        return loss_pi, pi_info
    
    def compute_loss_log_alpha(mmd_entropy):
        if log_alpha < -5.0:
            loss_log_alpha = -log_alpha
        elif log_alpha > 5.0:
            loss_log_alpha = log_alpha
        else:
            loss_log_alpha = log_alpha * (beta - mmd_entropy)
        return loss_log_alpha

    def compute_loss_log_penalty(cost):
        if log_penalty< -5.0:
            loss_log_penalty = -log_penalty
        elif log_penalty > np.log(largest_penalty):
            loss_log_penalty = log_penalty
        else:
            loss_log_penalty = log_penalty * (largest_cost - cost)
        return loss_log_penalty

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    cq_optimizer = Adam(cq_params, lr=lr)
    log_alpha_optimizer = Adam([log_alpha], lr=lr)
    log_penalty_optimizer = Adam([log_penalty], lr=cost_lr)

    # Set up model saving
    logger.setup_pytorch_saver({'ac':ac.state_dict()})

    def smooth_set(t_targ, t, polyak=polyak):
        t_targ.mul_(polyak)
        t_targ.add_((1 - polyak) * t)

    def smooth_update(model, model_targ, polyak=polyak):
        for p, p_targ in zip(model.parameters(), model_targ.parameters()):
            smooth_set(p_targ.data, p.data)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        cq_optimizer.zero_grad()
        loss_q, loss_cq, q_info = compute_loss_q(data)
        loss_q.backward()
        loss_cq.backward()
        q_optimizer.step()
        cq_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), LossCQ=loss_cq.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False
        for p in cq_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        for p in q_params:
            p.requires_grad = True
        for p in cq_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Auto tuning alpha
        if auto_alpha:
            log_alpha_optimizer.zero_grad()
            loss_log_alpha = compute_loss_log_alpha(pi_info["mmd_entropy"])
            loss_log_alpha.backward()
            log_alpha_optimizer.step()
        alpha = log_alpha.exp().detach()
        logger.store(alpha=alpha)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            smooth_update(ac.q1, ac_targ.q1, polyak)
            smooth_update(ac.q2, ac_targ.q2, polyak)

    def get_action(o, deterministic=False):
        # o = replay_buffer.obs_encoder(o)
        o = torch.FloatTensor(o.reshape(1, -1)).to(device)
        a = ac.act(o, deterministic, noise=noise)
        return a

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_cost, ep_len = test_env.reset(), False, 0, 0, 0
            while not(d or (ep_len == max_ep_len)):
                o, r, d, info = test_env.step(get_action(o, True) * act_limit)
                ep_ret += info.get('goal_met', 0.0)
                ep_cost += 1.0 if info.get('cost', -1.0) > 0.0 else 0.0
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpCost=ep_cost, TestEpLen=ep_len)
    
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_cost, ep_cost_sparse, ep_len = env.reset(), 0, 0, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 

        if t > start_steps:
            a = get_action(o, deterministic=False)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a * act_limit)
        c = info.get('cost', 0.0)
        ep_ret += info.get('goal_met', 0.0)
        ep_cost += c
        ep_cost_sparse += 1.0 if c > 0 else 0.0
        ep_len += 1


        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)

        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r*reward_scale, (c - cost_bias)*cost_scale, o2, d) 
 
        with torch.no_grad():
            obs_std = torch.FloatTensor(replay_buffer.obs_std).to(device)
            obs_mean = torch.FloatTensor(replay_buffer.obs_mean).to(device)
            smooth_set(ac.obs_std.data, obs_std)
            smooth_set(ac.obs_mean.data, obs_mean)
            smooth_set(ac_targ.obs_std.data, obs_std)
            smooth_set(ac_targ.obs_mean.data, obs_mean)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # auto penalty
            if auto_penalty:
                print("ep_cost = {}".format(ep_cost))
                for _ in range(50):
                    log_penalty_optimizer.zero_grad()
                    loss_log_penalty = compute_loss_log_penalty(ep_cost)
                    loss_log_penalty.backward()
                    log_penalty_optimizer.step()
            penalty = log_penalty.exp().detach()
            logger.store(penalty=penalty)

            logger.store(EpRet=ep_ret, EpCost=ep_cost_sparse, EpLen=ep_len)
            o, ep_ret, ep_cost, ep_cost_sparse, ep_len = env.reset(), 0, 0, 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            epoch = (t+1) // steps_per_epoch
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

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
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', average_only=True)
            logger.log_tabular('Q2Vals', average_only=True)
            logger.log_tabular('Q1ValsStd', average_only=True)
            logger.log_tabular('Q2ValsStd', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('LossCQ', average_only=True)
            logger.log_tabular('mmd_entropy', average_only=True)
            logger.log_tabular('alpha', average_only=True)
            logger.log_tabular('penalty', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # Normalize State
            print("obs_mean=" + str(ac.obs_mean))
            print("obs_std=" + str(ac.obs_std))

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

    gac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

#   python -m spinup.run gac_pytorch --env HalfCheetah-v2 --exp_name sac_HalfCheetahv2