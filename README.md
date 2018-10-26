## snake-ai-reinforcement
AI for Snake game trained from pixels using Deep Reinforcement Learning (DQN).

<p align="Center">
  <img src="https://cloud.githubusercontent.com/assets/2750531/24808769/cc825424-1bc5-11e7-816f-7320f7bda2cf.gif" width="300px">
  <img src="https://cloud.githubusercontent.com/assets/2750531/24810302/9e4d6e86-1bca-11e7-869b-fc282cd600bb.gif" width="300px">
</p>

We implemented a sarsa method, 4 dqn extensions and the combined dqn version based on the previous basic dpn engine which was implemented by Yuriyguts (2017) [4]. We also compared to the results metioned in paper "Rainbow: Combining Improvements in Deep Reinforcement Learning" by DeepMind.

## Methods Implemented
- Sarsa
  <p align="Center"><img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/saras.png" width="500px"></p>
- Double Q function
  <p align="Center"><img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/double_Q_function.png" width="600px"></p>
- Multi-step rewards
  <p align="Center"><img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/Multi_step_rewards.png" width="280px"></p>
- Prioritized replay
  <p align="Center"><img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/prioritized_replay.png" width="600px"></p>
- Dueling networks
  <p align="Center"><img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/dueling_networks.png" width="600px"></p>
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

## Method Analysis
<p align="Center">
  <img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/Fruits_eaten_by_different_methods.png" width="800px">
</p>
<p>
  Similar to the find out in the Deepmind paper[2], we find out most methods improve the learning ability on Deep Reinforcement Learning (DQN). 
</p>
<p>
  The multi-step stepsize extension may suffer from future rewards which slower the network learning progress. The combined method (combined with extension Double Q function, Multi-step rewards, Prioritized replay, Dueling networks) outperforms the basic dpn and achieves the best performance among all methods. 
</p>
<p>
  It is worth mentioning that we highly recommend using Dueling networks architecture. It improves the performance a lot while comparing the basic dpn only slight changes need to be modified in the output layers.
</p>

## Hyperparameter Tuning - Experience Replay Memory Size
<p align="Center">
  <img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/Fruits_eaten_with_increase_of_memory.png" width="800px">
</p>
Our model is based on one important assumption which is the process is a Markov process, the big memory size, where we sample the batches from, can help us get rid of the dependence of the data.

## Hyperparameter Tuning - Multi-step Stepsize
<p align="Center">
  <img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/Different_stepsizes_of_multi_step.png" width="800px">
</p>
<p>
  The performance decreases as the stepsize gets bigger. This maybe due to larger multi-step stepsize could hinder the learning ability of the neural network. With large step size, the neural network is forced to consider future rewards which might slower the learning progress.
</p>

## Visualizing the Value Function
<p align="Center">
  <img src="https://raw.githubusercontent.com/mingyip/snake-ai-reinforcement/master/result/Maximal_expected_reward.png" width="800px">
</p>
<p align="Left"> 
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

## Check Out Our Poster
<p>
  <img src="https://github.com/mingyip/snake-ai-reinforcement/blob/master/result/poster.jpg" alt="Rainbow dqn Poster" width="128" border="0">
</p>

[[image]](https://github.com/mingyip/snake-ai-reinforcement/blob/master/result/poster.jpg)
[[pdf]](https://github.com/mingyip/snake-ai-reinforcement/blob/master/result/poster.pdf)


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
