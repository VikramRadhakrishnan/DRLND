# Project 1: Navigation

## Introduction
In this project we make use of a game environment developed in [Unity](https://unity.com/) to train an intelligent agent. The environment contains yellow and blue bananas, and the goal of the agent is to navigate the environment and collect as many yellow bananas as it can, while avoiding the blue bananas. See the animation below, taken from the Udacity deep reinforcement learning git repository, for an example.  
![Bananas!](./bananas.gif)

## Environment details
The state space of the environment is a vector of length 37, which includes information about the agent's velocity in different directions as well as ray-based perception of objects in the agent's forward direction. The agent is capable of taking 4 actions - move forward (default), move backward, move left, and move right. Every time the agent moves over a yellow banana, it gets a reward of +1. A blue banana on the other hand gives a reward of -1. The environment is considered "solved" by the agent when the agent is able to get an average score of +13 over 100 consecutive iterations.

## Instructions
To run the "Report.ipynb" notebook, you need to have downloaded the Unity environnment (if you are running it on Linux like I have, then you can get the environment [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip). There are also some dependencies that must be installed and you can do so by following the instructions under the "Dependencies" heading at this [link](https://github.com/udacity/deep-reinforcement-learning). Once all dependencies are installed, simply open the Jupyter notebook titled Report.ipynb, and run all cells. This will train the agent from scratch. Alternately, you can grab the trained agent [here](./checkpoint.pth).

## Model description
I used a Deep Q Learning agent identical to the one used in the Udacity lesson. This agent has a fully connected network with two hidden layers, each of 64 neurons. The input to the network is the 37 dimension state space and the output is a vector of size 4 where each of the 4 outputs corresponds to an action. Actions are selected based on an epsilon greedy policy.

## Results
The agent was trained in roughly 500 episodes to solve the environment. Overall a very satisfactory result. One possible improvement or rather upgrade, would be to train the agent not from the 37 length vector of observations, but directly from the pixels on the screen. This would make use of a deep convolutional network to extract spatial information from the pixels, and then translate this into actions.
