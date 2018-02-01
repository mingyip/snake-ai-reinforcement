#!/usr/bin/env python3.6

""" Front-end script for training a Snake agent. """

import time
import json
import sys
import os
import noisy_dense

from keras import backend as K
from noisy_dense import NoiseDense
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=30000,
        help='The number of episodes to run consecutively.',
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
    sigma =0.5
    inp=Input(shape=(num_last_frames, ) + env.observation_shape)
 #   model = Sequential()

    # Convolutions.
    out=Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first', activation='relu')(inp)
    out = Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first', activation='relu')(out)

    # Dense layers.
    out = Flatten()(out)
    outWithoutNoise1 = Dense(512)(out)

    one_input1 = np.ones((512, 2306))
    one_input2 = np.ones((3,512))
    one_input1 = K.variable(one_input1)
    one_input2 = K.variable(one_input2)
    one_input1=Input(tensor=one_input1)
    one_input2=Input(tensor=one_input2)
    out_noise =  noise_weight1=Dense(512)(out)
    n1 = np.random.normal(0, sigma, (512))
    n2 = np.random.normal(0, sigma, (3,512))
    n1 = K.variable(n1)
    n2 = K.variable(n2)
    n1 = Input(tensor=n1)
    n2 = Input(tensor=n2)

    noise1 = merge([n1,out_noise], mode= "mul")
    
    added = keras.layers.Add()([outWithoutNoise1, noise1])
    actions = Dense(3)(out)



 #   model.add(Dense(256))
 #   model.add(Activation('relu'))
 #   model.add(Dense(env.num_actions))
    model = Model(inputs = inp, outputs = actions)
    model.summary()
    model.compile(RMSprop(), 'MSE')

    return model


def main():
    # Create a folder for data output.
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_path = os.path.join('outputs', str(timestamp))
    os.makedirs(output_path)

    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level, output_path)
    model = create_dqn_model(env, num_last_frames=4)


    agent = DeepQNetworkAgent(
        model=model,
        memory_size=-1,
        num_last_frames=model.input_shape[1],
        output=output_path
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.95
    )


if __name__ == '__main__':
    main()
