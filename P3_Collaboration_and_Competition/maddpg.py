from ddpg_agent import Agent
import numpy as np
import random
import torch
import copy
from collections import namedtuple, deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size

N_LEARN_UPDATES = 1    # number of learning updates
N_TIME_STEPS = 1       # every n time steps the network is updated

class MADDPG:
    """Instantiate the agents and ensure that they learn from a common replay buffer."""

    def __init__(self, num_agents, state_size, action_size, random_seed):
        super(MADDPG, self).__init__()
        
        self.seed = random_seed
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared replay memory
        self.memory = ReplayBuffer(num_agents, BUFFER_SIZE, BATCH_SIZE, self.seed)
        
        self.agents = [Agent(state_size, action_size, self.memory, (x+1) * random_seed) for x in range(num_agents)]

    def step(self, time_step, states, actions, rewards, next_states, dones):
        """Save experience in replay memory and train each agent from shared memories"""
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # Only update after collecting memories for n time steps
        if time_step % N_TIME_STEPS != 0:
            return
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            
            # Train the network for a number of epochs specified by the parameter
            for i in range(N_LEARN_UPDATES):
                for agent in self.agents:
                    agent.step()

    def act(self, states, add_noise=True):
        """Returns actions for given state per agent"""
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions

    def reset(self):        
        for agent in self.agents:
            agent.reset()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_agents, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Buffer contains n-dimensional s, a, r, s' and d where n = number of agents
        # When sampling from buffer reshape s, a, r, s' and d to match dimensions of a single agent
        states = torch.from_numpy(np.vstack([[e.state[j] for j in range(self.num_agents)] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([[e.action[j] for j in range(self.num_agents)] for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([[[e.reward[j]] for j in range(self.num_agents)] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([[e.next_state[j] for j in range(self.num_agents)] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([[[e.done[j]] for j in range(self.num_agents)] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
