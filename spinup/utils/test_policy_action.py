import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

device = torch.device('cuda')

def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action

def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname).to(device)

    # make function for producing an action given a single state
    def get_action(o):
        with torch.no_grad():
            o = torch.FloatTensor(o.reshape(1, -1)).to(device)
            action = model.act(o, deterministic)
        if 'gac' not in fpath:
            action = action[0]
        return action

    return get_action

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    axis_bound = env.action_space.high[0] + 0.01

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.reshape(-1)
    dots = [ax.plot([], [], 'bo', alpha=0.02)[0] for ax in axs]

    def init():
        for ax in axs:
            ax.set_xlim(-axis_bound, axis_bound)
            ax.set_ylim(-axis_bound, axis_bound)
            ax.grid()

    def gen_dot():
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        logger = EpochLogger()
        while n < num_episodes:
            if render:
                env.render()
                time.sleep(1e-3)

            a = get_action(o)
            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            actions = []
            for _ in range(1000):
                actions.append(get_action(o))
            yield np.array(actions)

            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                n += 1

        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.dump_tabular()

    def update_dot(actions):
        dots[0].set_data(actions[:, 0], actions[:, 1])
        dots[1].set_data(actions[:, 0], actions[:, 2])
        dots[2].set_data(actions[:, 1], actions[:, 2])
        dots[3].set_data(actions[:, 3], actions[:, 4])
        dots[4].set_data(actions[:, 3], actions[:, 5])
        dots[5].set_data(actions[:, 4], actions[:, 5])
        return dots
        
    ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 20, init_func=init)
    print('generative')
    ani.save('./action.gif', writer='pillow', fps=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))