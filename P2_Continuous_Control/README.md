# Project 2: Continuous Control

## Introduction
In this project we make use of a game environment developed in [Unity](https://unity.com/) to train a double jointed robotic arm to move to a target location. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The agent's goal is to maintain its position at the target location for as many time steps as possible. See the animation below, taken from the Udacity deep reinforcement learning git repository, for an example.  
![Trained agent](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

## Environment details
The state space of the environment is a vector of length 33 for each arm in the environment, which encodes the position, rotation, velocity, and angular velocities of that arm. I chose to solve the version of the environment with 20 arms. For each arm, there are is a vector of four possible actions corresponding to torque applicable to two joints. Every entry in the action vector is a number between -1 and 1. If the hand of an arm is moved to the target location, the corresponding agent receives a reward of +0.1 at each timestep the hand is in the target location. The environment is considered "solved" by the agent when the agent is able to get an average score of +30.0 over 100 consecutive iterations.

## Instructions
To run the "Report.ipynb" notebook, you need to have downloaded the Unity environnment (if you are running it on Linux like I have, then you can get the environment [here](hhttps://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip). There are also some dependencies that must be installed and you can do so by following the instructions under the "Dependencies" heading at this [link](https://github.com/udacity/deep-reinforcement-learning). Once all dependencies are installed, simply open the Jupyter notebook titled Report.ipynb, and run all cells. This will train the agent from scratch. Alternately, you can grab the trained actor agent [here](./checkpoint_actor.pth).

## Model description
I used a Deep Deterministic Policy Gradient agent inspired by the one used in the Udacity lesson with the inverted pendulum environment. This agent makes use of two fully connected networks, an actor and a critic. I initialized all the model parameters and hyperparameters according to the original paper on [Deep Deterministic Policy Gradients](https://arxiv.org/pdf/1509.02971.pdf). The actor network takes in the state and outpus an action. It has two fully connected layers of 400 and 300 units each. The critic network takes the state and action as inputs and outputs a Q value. This network has a fully connected layer of size 400 after the state input, followed by concatenating this output with the action input and passing that through a fully connected layer of size 300 before going to the input node. The network parameters and hyperparameters are detailed below:

| Parameter  | Value |
| ------------- | ------------- |
| Actor learning rate  | 1e-4   |
| Critic learning rate  | 1e-3  |
| Replay buffer size | 1e5 |
| Batch size | 128 |
| Gamma | 0.99 |
| Tau for soft updates | 1e-3 |
| OU noise parameters | mu = 0, theta = 0.15, sigma = 0.2 |

During the training process, I followed a hint given in the Udacity classroom, where I updated the networks once every 20 steps, and trained them for 10 epochs during the update step.

## Results
The agent was trained in 276 episodes to solve the environment. Looking at the evolution of the average score with respect to training time, it looks like a smooth, almost linear increase. This makes me think that I can speed up the training by playing with the learning rates a bit, and also perhaps reducing the number of neurons in the hidden layers. Also, increasing the batch size might help converge faster.
