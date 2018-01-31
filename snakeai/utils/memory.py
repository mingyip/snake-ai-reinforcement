import collections
import random

import numpy as np


class ExperienceReplay(object):
    """ Represents the experience replay memory that can be randomly sampled. """

    def __init__(self, input_shape, num_actions, memory_size=100):
        """
        Create a new instance of experience replay memory.
        
        Args:
            input_shape: the shape of the agent state.
            num_actions: the number of actions allowed in the environment.
            memory_size: memory size limit (-1 for unlimited).
        """
        self.memory = collections.deque()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.memory_size = memory_size

    def reset(self):
        """ Erase the experience replay memory. """
        self.memory = collections.deque()

    def remember(self, state, action, reward, state_next, is_episode_end):
        """
        Store a new piece of experience into the replay memory.
        
        Args:
            state: state observed at the previous step.
            action: action taken at the previous step.
            reward: reward received at the beginning of the current step.
            state_next: state observed at the current step. 
            is_episode_end: whether the episode has ended with the current step.
        """
        memory_item = np.concatenate([
            state.flatten(),
            np.array(action).flatten(),
            np.array(reward).flatten(),
            state_next.flatten(),
            1 * np.array(is_episode_end).flatten()
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

    def remember(self, state, action, reward, state_next,action_next, is_episode_end):
        """
        Store a new piece of experience into the replay memory.
        
        Args:
            state: state observed at the previous step.
            action: action taken at the previous step.
            reward: reward received at the beginning of the current step.
            state_next: state observed at the current step. 
            is_episode_end: whether the episode has ended with the current step.
        """
        memory_item = np.concatenate([
            state.flatten(),
            np.array(action).flatten(),
            np.array(reward).flatten(),
            state_next.flatten(),
            1 * np.array(is_episode_end).flatten(),
            np.array(action_next).flatten()
        ])
        self.memory.append(memory_item)
        if 0 < self.memory_size < len(self.memory):
            self.memory.popleft()

  

    def get_batch(self, model, batch_size,exploration_rate, discount_factor=0.9, method='dqn'):
        """ Sample a batch from experience replay. """
        batch_size = min(len(self.memory), batch_size)
        experience = np.array(random.sample(self.memory, batch_size))
        input_dim = np.prod(self.input_shape)

        # Extract [S, a, r, S', end] from experience.
        states = experience[:, 0:input_dim]
        actions = experience[:, input_dim]
        rewards = experience[:, input_dim + 1]
        states_next = experience[:, input_dim + 2:2 * input_dim + 2]
        episode_ends = experience[:, 2 * input_dim + 2]
        #action_next = experience[:,2 * input_dim + 3]

        # Reshape to match the batch structure.
        states = states.reshape((batch_size, ) + self.input_shape)
        actions = np.cast['int'](actions)
        #action_next = np.cast['int'](action_next)
        rewards = rewards.repeat(self.num_actions).reshape((batch_size, self.num_actions))
        states_next = states_next.reshape((batch_size, ) + self.input_shape)
        episode_ends = episode_ends.repeat(self.num_actions).reshape((batch_size, self.num_actions))

        X = np.concatenate([states, states_next], axis=0)
        y = model.predict(X)
   #     print('---------------------')
        #print(y.shape)

       # print(action_next)
        # Predict future state-action values.
        if method == 'sarsa':
            
            y1 = y[batch_size:]

            action_next = y1.argmax(axis=1)
   #         print(action_next)
            nr_random = round(batch_size*exploration_rate)
            indices_for_random = np.random.choice(batch_size, size=nr_random) 
            action_next[indices_for_random]=np.random.randint(3, size=nr_random)
    #        print(action_next)
     #       print(indices_for_random)

            Q_next = np.choose(action_next, y1.T).repeat(self.num_actions)
            Q_next = Q_next.reshape((batch_size, self.num_actions))
            #print(Q_next)

        elif method == 'ddqn':
            y = y[batch_size:,:]
            a = model.predict(states_next)
            a = a.argmax(axis=1)
            Q_next = np.choose(a, y.T).repeat(self.num_actions)
            Q_next = Q_next.reshape((batch_size, self.num_actions))
        else:
            # qlearning
            Q_next = np.max(y[batch_size:], axis=1).repeat(self.num_actions).reshape((batch_size, self.num_actions))
            
        delta = np.zeros((batch_size, self.num_actions))
        delta[np.arange(batch_size), actions] = 1
        #print(y.shape )
        #print('----------')
        #we use y[:batch_size] here but set #y = y[batch_size:,:] above
        targets = (1 - delta) * y[:batch_size] + delta * (rewards + discount_factor * (1 - episode_ends) * Q_next)
        return states, targets
