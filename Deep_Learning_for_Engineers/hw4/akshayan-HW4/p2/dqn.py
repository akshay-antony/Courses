import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

# The starter code follows the tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# we recommend you going through the tutorial before implement DQN algorithm


# define environment, please don't change 
env = gym.make('CartPole-v1')

# define transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory_buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory_buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory_buffer, batch_size)

    def __len__(self):
        return len(self.memory_buffer)

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        in_dim: dimension of states
        out_dim: dimension of actions
        """
        super(DQN, self).__init__()
        self.layer_list = nn.ModuleList()
        list_sizes = [32, 64, 128, 256, 256]
        self.layer_list.append(nn.Sequential(nn.Linear(in_dim, list_sizes[0]),
                                             nn.BatchNorm1d(list_sizes[0])))
        
        for i in range(len(list_sizes) - 1):
            self.layer_list.append(nn.Sequential(nn.Linear(list_sizes[i], list_sizes[i+1]),
                                                 nn.BatchNorm1d(list_sizes[i+1])))
        self.layer_list.append(nn.Linear(list_sizes[-1], out_dim))
        self.act = nn.ReLU()

    def forward(self, x):
        # forward pass
        out = x.to(device)
        for i in range(len(self.layer_list)):
            if i != len(self.layer_list)-1:
                out = self.act(self.layer_list[i](out))
            else:
                out = self.layer_list[i](out)
        return out
    
# hyper parameters you can play with
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer = optim.Adam(policy_net.parameters(), lr=0.03)
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.0025)   
memory = ReplayMemory(MEMORY_CAPACITY)
steps_done = 0

def select_action(state):
    global steps_done
    # given state, return the action with highest probability on the prediction of DQN model
    # you are recommended to also implement a soft-greedy here
    sample = random.random()
    eps_thresh = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_thresh:
        with torch.no_grad():
            policy_net.eval()
            action_return = policy_net(state).max(1)[1].view(1, 1)
            policy_net.train()
            return action_return
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # optimize the DQN model by sampling a batch from replay buffer
    batch = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*batch))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_states = torch.cat([s for s in batch.next_state if s is not None])
    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)

    q_val_pred = policy_net(states).gather(1, actions)
    future_state_q_values = torch.zeros(BATCH_SIZE, device=device)
    future_state_q_values[non_final_mask] = target_net(non_final_states).max(1)[0].detach()
    target_state_q_values = (future_state_q_values * GAMMA) + rewards
    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(q_val_pred.squeeze(), target_state_q_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 150  
episode_durations = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.from_numpy(state).float().view(1, -1)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        new_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # # Observe new state
        if not done:
            next_state = torch.from_numpy(new_state).float().view(1, -1)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            print("Episode: {}, duration: {}".format(i_episode, t+1))
            break
    
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), "q2_policy_net.pth")
    torch.save(target_net.state_dict(), "q2_target_net.pth")
# plot time duration
plt.figure()
plt.plot(np.arange(len(episode_durations)), episode_durations)
plt.show()
plt.savefig('books_read.png')

# visualize 
for i in range(10):
    state = env.reset()
    state = torch.from_numpy(state).float().view(1, -1)
    for t in count():
        env.render()

        # Select and perform an action
        action = select_action(state)
        new_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward])

        # Observe new state
        if not done:
            next_state = torch.from_numpy(new_state).float().view(1, -1)
        else:
            next_state = None

        # Move to the next state
        state = next_state

        if done:
            episode_durations.append(t + 1)
            print("Duration:", t+1)
            break

env.close()
