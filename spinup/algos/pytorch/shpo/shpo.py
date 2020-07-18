from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.shpo.core as core
from spinup.utils.logx import EpochLogger

def shpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), 
    seed=0, device='cpu', steps_per_epoch=4000, epochs=50, replay_size=1000000, 
    gamma=0.99, polyak=0.005, polyak_pi = 0.0, lr=1e-3, 
    batch_size=100,expand_batch=100,
    start_steps=10000, update_after=10000, num_test_episodes=10, 
    per_update_steps_for_actor=100, per_update_steps_for_critic=50, 
    cg_iters=10, max_ep_len=1000, 
    logger_kwargs=dict(), save_freq=1, algo='shpo'):
    """
    Sinkhorn Policy Optimization (SHPO)

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

        polyak_pi (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        batch_size (int): Minibatch size for Critic.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # ====== All About Init ===============================================================
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
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)

    # Experience buffer
    replay_buffer = core.ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    # ===== End Of Init =========================================================================    

    # ===== Critic Loss =========================================================================
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        o  = torch.FloatTensor(o).to(device)
        a  = torch.FloatTensor(a).to(device)
        r  = torch.FloatTensor(r).to(device)
        o2 = torch.FloatTensor(o2).to(device)
        d  = torch.FloatTensor(d).to(device)

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

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
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info
    
    def update_critic(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

            for p, p_targ in zip(ac.pi.parameters(), ac_targ.pi.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak_pi)
                p_targ.data.add_(polyak_pi * p.data)
    # ===== End Of Critic Loss ============================================================================

    # ===== Update Actor ==================================================================================
    def compute_loss_pi(data):
        o = data['obs']
        o = torch.FloatTensor(o).to(device)

        o2 = o.repeat(expand_batch, 1)
        a2 = ac.pi(o2)
        q1_pi = ac.q1(o2, a2)
        q2_pi = ac.q2(o2, a2)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = -q_pi.mean()
        return loss_pi

    def update_actor(data):
        for p in q_params:
            p.requires_grad = False

        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        logger.store(LossPi=loss_pi.item())

        """
        # ??? I am not sure: Do I need zero_grad()?
        loss_pi = compute_loss_pi(data)
        grads = torch.autograd.grad(loss_pi, ac.pi.parameters())
        grads_vector = torch.cat([grad.view(-1) for grad in grads]).data

        def get_Hx(x):
            # Require New Method.

        invHg = core.cg(get_Hx, loss_grad, cg_iters)
        # fullstep = ???

        with torch.no_grad():
            prev_params = core.get_flat_params_from(ac.pi)
            # new_params = ???
            # core.set_flat_params_to(ac.pi, new_params)
        """

        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item())
    # ===== End Of Actor ==================================================================================

    # ===== Start Training ================================================================================
    def get_action(o, deterministic=False):
        # o = replay_buffer.obs_encoder(o)
        o = torch.FloatTensor(o.reshape(1, -1)).to(device)
        a = ac.act(o, deterministic)
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
        if t >= update_after and (t+1) % steps_per_epoch == 0:
            for j in range(per_update_steps_for_critic):
                data = replay_buffer.sample_batch(batch_size)
                update_critic(data);

            for j in range(per_update_steps_for_actor):
                data = replay_buffer.sample_recently(steps_per_epoch)
                update_actor(data);

        # End of epoch handling
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
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            # Normalize State
            print("obs_mean="+str(replay_buffer.obs_mean))
            print("obs_std=" +str(replay_buffer.obs_std))
    # ====== End Training ==================================================================================
        