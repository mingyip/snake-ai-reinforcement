#!/usr/bin/env python3.6

""" Front-end script for training a Snake agent. """

import time
import json
import sys
import os

from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
from keras.optimizers import *

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser
from config import Config


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=False,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=False,
        type=int,
        help='The number of episodes to run consecutively.',
    )
    parser.add_argument(
        '--model',
        required=False,
        type=str,
        help='The pretrained model to continue training.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename, output):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, output=output, verbose=1)


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.
    
    Args:
        env: an instance of Snake environment. 
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """

    model = Sequential()

    # Convolutions.
    model.add(Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first',
        input_shape=(num_last_frames, ) + env.observation_shape
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first'
    ))
    model.add(Activation('relu'))

    # Dense layers.
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env.num_actions))

    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model


def main():
    # Create a folder for data output.
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_path = os.path.join('outputs', str(timestamp))
    os.makedirs(output_path)

    # Handle input params. load parsed args if specified by users
    # otherwise load from config file
    parsed_args = parse_command_line_args(sys.argv[1:])
    level = parsed_args.level if parsed_args.level else Config.LEVEL
    num_episodes = parsed_args.num_episodes if parsed_args.num_episodes else Config.NUM_EPISODES


    env = create_snake_environment(level, output_path)
    if parsed_args.model:
        model = load_model(parsed_args.model)
    elif Config.USE_PRETRAINED_MODEL:
        model = load_model(Config.PRETRAINED_MODEL)
    else:
        model = create_dqn_model(env, num_last_frames=Config.NUM_LAST_FRAMES)


    agent = DeepQNetworkAgent(
        model=model,
        memory_size=Config.MEMORY_SIZE,
        num_last_frames=model.input_shape[1],
        output=output_path
    )
    agent.train(
        env,
        batch_size=Config.BATCH_SIZE,
        num_episodes=num_episodes,
        checkpoint_freq=num_episodes // 10,
        discount_factor=Config.DISCOUNT_FACTOR,
        exploration_range=(Config.MAX_EXPLORATION, Config.MIN_EXPLORATION)
    )


if __name__ == '__main__':
    main()
