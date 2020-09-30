import gym
import safety_gym
import torch
from core import MLPActorCritic as actor_critic

import time

def test(device, env, model_file, epoch=10, max_ep_len=1000, render=False, hidden_sizes=[256,256]):
    device = torch.device(device)

    env = gym.make(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    ac = actor_critic(obs_dim, act_dim, hidden_sizes).to(device)

    model_parameters = torch.load(model_file)
    ac.load_state_dict(model_parameters['ac'])

    for p in ac.parameters():
        p.requires_grad = False

    def get_action(o, deterministic=False):
        o = torch.FloatTensor(o.reshape(1, -1)).to(device)
        a = ac.act(o, deterministic)
        return a

    total_ret, total_cost = 0, 0
    for t in range(epoch):
        o, ep_ret, ep_cost = env.reset(), 0, 0
        for _ in range(max_ep_len):
            if render:
                env.render()
                time.sleep(1e-3)

            a = get_action(o, deterministic=False)

            o, r, d, info = env.step(a * act_limit)
            c = info.get('cost', 0.0)
            ep_ret += info.get('goal_met', 0.0)
            ep_cost += c
        
        print("Epoch {}: ep_ret = {}, ep_cost = {}".format(t, ep_ret, ep_cost))
        total_ret += ep_ret
        total_cost += ep_cost

    print("Total : ep_ret = {}, ep_cost = {}".format(total_ret/epoch, total_cost/epoch))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--model-file', type=str)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    test(args.device, args.env, args.model_file, args.epochs, render=args.render)