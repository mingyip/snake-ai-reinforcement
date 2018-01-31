import collections
import numpy as np
import time
import os

from snakeai.agent import AgentBase
from snakeai.utils.memory import ExperienceReplay
from contextlib import redirect_stdout

class DeepQNetworkAgent(AgentBase):
    """ Represents a Snake agent powered by DQN with experience replay. """

    def __init__(self, model, num_last_frames=4, memory_size=1000, output="."):
        """
        Create a new DQN-based agent.
        
        Args:
            model: a compiled DQN model.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited). 
            output (str): folder path to output model files.
        """
        assert model[0].input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model[0].output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'

        self.model = model
        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay((num_last_frames,) + model[0].input_shape[-2:], model[0].output_shape[-1], memory_size)
        self.frames = None
        self.output = output
        self.num_frames = 0
        self.num_trained_frames = 0

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.frames = None

    def get_last_frames(self, observation):
        """
        Get the pixels of the last `num_last_frames` observations, the current frame being the last.
        
        Args:
            observation: observation at the current timestep. 

        Returns:
            Observations for the last `num_last_frames` frames.
        """
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0)

    def train(self, env, num_episodes=1000, batch_size=50, discount_factor=0.9, checkpoint_freq=None,
              exploration_range=(1.0,0.1), exploration_phase_size=0.5, method='dqn'):
        """
        Train the agent to perform well in the given Snake environment.
        
        Args:
            env:
                an instance of Snake environment.
            num_episodes (int):
                the number of episodes to run during the training.
            batch_size (int):
                the size of the learning sample for experience replay.
            discount_factor (float):
                discount factor (gamma) for computing the value function.
            checkpoint_freq (int):
                the number of episodes after which a new model checkpoint will be created.
            exploration_range (tuple):
                a (max, min) range specifying how the exploration rate should decay over time. 
            exploration_phase_size (float):
                the percentage of the training process at which
                the exploration rate should reach its minimum.
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # Calculate the constant exploration decay speed for each episode.
        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (num_episodes * exploration_phase_size))
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0
            model_to_udate = np.random.randint(0, 2) if method == 'ddqn' else 0

            # Observe the initial state.
            state = self.get_last_frames(timestep.observation)

            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    q = self.model[model_to_udate].predict(state)
                    action = np.argmax(q[0])

                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                # Remember a new piece of experience.
                reward = timestep.reward
                state_next = self.get_last_frames(timestep.observation)

                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action_next = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    q = self.model[model_to_udate].predict(state_next)
                    action_next = np.argmax(q[0])

                game_over = timestep.is_episode_end
                experience_item = [state, action, reward, state_next, action_next, game_over]
                self.memory.remember(*experience_item)
                state = state_next
 
                # Sample a random batch from experience.
                batch = self.memory.get_batch(
                    model=self.model,
                    batch_size=batch_size,
                    discount_factor=discount_factor,
                    method=method,
                    model_to_udate=model_to_udate
                )
                
                # Learn on the batch.
                if batch:
                    inputs, targets = batch
                    self.num_trained_frames += targets.size
                    loss += float(self.model[model_to_udate].train_on_batch(inputs, targets))

            if checkpoint_freq and (episode % checkpoint_freq) == 0:
                self.model[0].save(f'{self.output}/dqn-{episode:08d}.model')
                self.evaluate(env, trained_episode=episode, num_test_episode=15)

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay
            
            self.num_frames += env.stats.timesteps_survived

            summary = 'Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Reward {:4d} | ' + \
                      'Memory {:6d} | Total Timesteps {:6d} | Trained Frames{:9d}'

            print(summary.format(
                episode + 1, num_episodes, loss, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
                len(self.memory.memory), self.num_frames, self.num_trained_frames
            ))
            with open(f'{self.output}/training-log.txt', 'a') as f:
                with redirect_stdout(f):
                    print(summary.format(
                        episode + 1, num_episodes, loss, exploration_rate,
                        env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
                        len(self.memory.memory), self.num_frames, self.num_trained_frames
                    ))
            f.close()

        self.model[0].save(f'{self.output}/dqn-final.model')
        self.evaluate(env, trained_episode=episode, num_test_episode=15)
        print('Training End - saved to ' + str(self.output))

    def act(self, observation, reward):
        """
        Choose the next action to take.
        
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.

        Returns:
            The index of the action to take next.
        """
        state = self.get_last_frames(observation)
        q = self.model[0].predict(state)[0]
        return np.argmax(q)

    def evaluate(self, env, trained_episode, num_test_episode):
        """
        Play a set of episodes using the specified Snake agent.
        Use the non-interactive command-line interface and print the summary statistics afterwards.
        
        Args:
            env: an instance of Snake environment.
            trained_episode (int): trained episodes.
            num_test_episode (int): the number of episodes to run.
        """

        fruit_stats = []
        timestep_stats = []
        reward_stats = []

        print()
        print('Playing:')

        for episode in range(num_test_episode):
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False

            while not game_over:
                action = self.act(timestep.observation, timestep.reward)
                env.choose_action(action)
                timestep = env.timestep()
                game_over = timestep.is_episode_end

            fruit_stats.append(env.stats.fruits_eaten)
            timestep_stats.append(env.stats.timesteps_survived)
            reward_stats.append(env.stats.sum_episode_rewards)

            summary = 'Episode {:3d} / {:3d} | Timesteps {:4d} | Fruits {:2d} | Reward {:3d}'
            print(summary.format(episode + 1, num_test_episode, env.stats.timesteps_survived, +\
            env.stats.fruits_eaten, env.stats.sum_episode_rewards))

        print('Fruits eaten {:.1f} +/- stddev {:.1f}'.format(np.mean(fruit_stats), np.std(fruit_stats)))
        print('Reward {:.1f} +/- stddev {:.1f}'.format(np.mean(reward_stats), np.std(reward_stats)))
        print()

        with open(f'{self.output}/training-stat.txt', 'a') as f:
                with redirect_stdout(f):
                    summary = 'Episode {:7d} | Average Timesteps {:4.0f} | Average Fruits {:.1f} | Average Reward {:.1f}' 
                    print(summary.format(trained_episode, np.mean(timestep_stats), np.mean(fruit_stats), np.mean(reward_stats)))
