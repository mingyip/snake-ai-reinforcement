
we got the code for the enviroment from https://github.com/YuriyGuts/snake-ai-reinforcement
we highly modified utils/memory.py, agent/dqn.py and train.py to implement DDQN, SARSA, multistep, priorized replay and dueling network.
we also created a logging system and made changes to the gameplay and enviroment
we did hyperparametersearch mostly for the memory size of experience replay, stepsize of multistep and different decays of the exploration rate



## snake-ai-reinforcement
AI for Snake game trained from pixels using Deep Reinforcement Learning (DQN).

Contains the tools for training and observing the behavior of the agents, either in CLI or GUI mode.

<img src="https://cloud.githubusercontent.com/assets/2750531/24808769/cc825424-1bc5-11e7-816f-7320f7bda2cf.gif" width="300px"><img src="https://cloud.githubusercontent.com/assets/2750531/24810302/9e4d6e86-1bca-11e7-869b-fc282cd600bb.gif" width="300px">

We implemented a sarsa method, 4 dpn extensions and the combined version. and compared to the results metioned in paper "Rainbow: Combining Improvements in Deep Reinforcement Learning" by DeepMind.

## Methods Implemented
- Sarsa
- Double Q function
- Multi-step rewards
- Prioritized replay
- Dueling networks
- Hybrid Method

## Requirements
All components have been written in Python 3.6. Training on GPU is supported but disabled by default. If you have CUDA and would like to use a GPU, use the GPU version of TensorFlow by changing `tensorflow` to `tensorflow-gpu` in the requirements file.

To install all Python dependencies, run:
```
$ make deps
```

## Pre-Trained Models
You can find a few pre-trained DQN agents on the [Releases](https://github.com/YuriyGuts/snake-ai-reinforcement/releases) page. Pass the model file to the `play.py` front-end script (see `play.py -h` for help).

* `dqn-10x10-blank.model`
  
  An agent pre-trained on a blank 10x10 level (`snakeai/levels/10x10-blank.json`).
  
* `dqn-10x10-obstacles.model`

  An agent pre-trained on a 10x10 level with obstacles (`snakeai/levels/10x10-obstacles.json`).


## Training a DQN Agent
To train an agent using the default configuration, run:
```
$ make train
```

The trained model will be checkpointed during the training and saved as `dqn-final.model` afterwards.

Run `train.py` with custom arguments to change the level or the duration of the training (see `train.py -h` for help).

## Playback

The behavior of the agent can be tested either in batch CLI mode where the agent plays a set of episodes and outputs summary statistics, or in GUI mode where you can see each individual step and action.

To test the agent in batch CLI mode, run the following command and check the generated **.csv** file:
```
$ make play
```

To use the GUI mode, run:
```
$ make play-gui
```

To play on your own using the arrow keys (I know you want to), run:
```
$ make play-human
```

## Hyperparameter Tuning - Multi-step Stepsize
<p align="Center">
  <img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/Different_stepsizes_of_multi_step.png" width="800px">
The performance decreases as the stepsize gets bigger. This maybe due to larger multi-step stepsize could hinder the learning ability of the neural network. With larger stepsize, the neural network is foreced to consider future rewards which might slower the learning progress. 

## Visualizing the Value Function
<p align="Center">
  <img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/Maximal_expected_reward.png" width="800px">
</p>
<p> 
  The graph shows how the expected maximal reward increases or decreases during one game. Note that every time the expected maximal reward goes down, it means that the snake ate a fruit at the last step. Because the fruit is further away at the next step the expected reward goes down. 
</p>

## Findings
<dl>
  <li>Method</li>
    <ul>
      <li>The combination of DDQN, dueling networks, prioritized replay and multi-step performed the best.</li>
      <li>The performance SARSA and multi-step was worse than original DQN.</li>
    </ul>
  <li>Memory</li>
    <ul>
      <li>The more memory we used for Experience Replay the better the performance.</li>
    </ul>
  <li>Multistep</li>
    <ul>
      <li>The performance decreases as the stepsize gets bigger.</li>
    </ul>
</dl> 

## References
<p> [1] Fuertes, R. A (2017). Machine Learning Agents: Making Video Games Smarter With AI. Retrieved
January 15, 2018 from https://edgylabs.com/machine-learning-agents-making-video-
games-smarter-with-ai</p>

<p> [2] Hessel, M., Modayil, J. & Hasselt, H. V. (2017). Rainbow: Combining Improvements in Deep
Reinforcement Learning. Retrieved January 10, 2018 from
https://arxiv.org/pdf/1710.02298.pdf </p>

<p> [3] Mnih, V., Kavukcuoglu, K. & Silver, D. (2013). 1312.5602v1.pdf. Retrieved January 10, 2018 from
https://arxiv.org/pdf/1312.5602v1.pdf </p>

<p> [4] Yuriyguts (2017). GitHub - YuriyGuts/snake-ai-reinforcement: AI for Snake game trained from pixels
using Deep Reinforcement Learning (DQN) . Retrieved January15, 2018 from
https://github.com/YuriyGuts/snake-ai-reinforcement </p>
