import copy
from enum import Enum

import numpy as np

import spinup.algos.pytorch.gac_her.core as core
import torch
# Hindsight Experience Replay

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
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

    def sample_batch(self, batch_size=100):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_encoder(self.obs_buf[idxs]),
                     obs2=self.obs_encoder(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    def obs_encoder(self, o):
        return ((np.array(o) - self.obs_mean)/(self.obs_std + 1e-8)).clip(-self.obs_limit, self.obs_limit)

class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM
}


class HindsightExperienceReplayWrapper(object):
    """
    Wrapper around a replay buffer in order to use HER.
    This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.

    :param replay_buffer: (ReplayBuffer)
    :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
    :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
        the goals for the artificial transitions.
    :param wrapped_env: (HERGoalEnvWrapper) the GoalEnv wrapped using HERGoalEnvWrapper,
        that enables to convert observation to dict, and vice versa
    """

    def __init__(self, wrapped_env, replay_buffer, n_sampled_goal=4, 
            goal_selection_strategy=KEY_TO_GOAL_STRATEGY['future']):

        super(HindsightExperienceReplayWrapper, self).__init__()

        assert isinstance(goal_selection_strategy, GoalSelectionStrategy), "Invalid goal selection strategy," \
                                                                           "please use one of {}".format(
            list(GoalSelectionStrategy))

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = wrapped_env
        # Buffer for storing transitions of the current episode
        self.episode_transitions = []
        self.replay_buffer = replay_buffer

    def store(self, obs, act, rew, next_obs, done, info):
        """
        add a new transition to the buffer

        :param obs: (np.ndarray) the last observation
        :param act: ([float]) the action
        :param rew: (float) the reward of the transition
        :param next_obs: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        :param info: (dict) is the info of the transition
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append((obs, act, rew, next_obs, done, info))
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def sample_batch(self, batch_size=100):
        return self.replay_buffer.sample_batch(batch_size)

    def __len__(self):
        return len(self.replay_buffer)

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.

        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal']

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.

        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs, act, rew, obs_next, done, info= transition

            # Add to the replay buffer
            self.replay_buffer.store(obs, act, rew, obs_next, done)

            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.episode_transitions) - 1 and
                    self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                # Copy transition to avoid modifying the original one
                obs, act, rew, next_obs, done, info = copy.deepcopy(transition)

                # Convert concatenated obs to dict, so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # Update the desired goal in the transition
                obs_dict['desired_goal'] = goal
                next_obs_dict['desired_goal'] = goal

                # Update the reward according to the new desired goal
                reward = self.env.compute_reward(next_obs_dict['achieved_goal'], goal, info)
                # Can we use achieved_goal == desired_goal?
                done = False

                # Transform back to ndarrays
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add artificial transition to the replay buffer
                self.replay_buffer.store(obs, act, rew, next_obs, done)
