# Project 3: Collaboration and Competition

## Introduction
In this project we train a pair of tennis-playing agents to keep a ball in the air for as long as possbible by hitting it back and forth to each other. The two agents control rackets to bounce a ball over a net. Both agents share a common goal of keeping the ball in play (i.e. not letting the ball hit the ground and not hitting the ball out of bounds) for as long as possible. See the animation below, taken from the Udacity deep reinforcement learning git repository, for a visual example of a pair of trained agents.  

![Trained agent](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

## Project details
In this environment, two separate agents play tennis with each other by bouncing a ball over a net. Each agent controls one tennis racket. Each agent has its own local observation space of 8 variables that correspond to th position and velocity of the ball and the racket. The goal of the game is to keep the ball in play for as long as possible. An agent receives a score of +0.1 every time it hits the ball over the net, and a score of -0.01 if the agent lets the ball hit the ground, or hits the ball out of the court. Each agent has two continuos actions, one corresponding to motion towards and away from the net, and the other corresponding to jumping. Both these action values are bounded between -1 and +1. After each episode, the score is calculated as the total undiscounted reward of the agent that scored the higher total score. The environment is considered solved when the average episode reward over 100 episodes is greater than or equal to 0.5.

## Getting Started
This project is the third and final project in the [Udacity Deep Reinforcement Learning Nanodegree programme](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). To run the "Report.ipynb" notebook, you need to have downloaded the Unity environnment (if you are running it on Linux like I have, then you can get the environment [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip). There are also some dependencies that must be installed and you can do so by following the instructions under the "Dependencies" heading at this [link](https://github.com/udacity/deep-reinforcement-learning). The files necessary to run this project are [Report.ipynb](./Report.ipynb), [model.py](./model.py), [ddpg_agent.py](ddpg_agent.py), and [maddpg.py](maddpg.py). Download all these files into the same directory.

## Instructions
Once all dependencies are installed and all files downloaded, simply open the Jupyter notebook titled Report.ipynb, and run all cells. This will train the agent from scratch. Alternately, you can grab the trained [agent 1](./agent1_checkpoint_actor.pth) and [agent 2](./agent2_checkpoint_actor.pth) and use them directly to play the game.
