import gym 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from model import * 
from replay_buffer import ReplayBuffer 
from visualize import * 
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo_name = 'SAC'       # Used for visualization // TODO implemenet ERE 
max_episodes = 2000     # after 200 episodes, SAC flatlines

gamma = 0.99            # Discount
α = 0.1                 # Entropy term
lr = 3e-4               # Determines how big of a gradient step to take when optimizing   TODO tinker with this
tau = 0.995             # Target smoothing coefficient? 

env = gym.make('Pendulum-v0')
replay_buffer = ReplayBuffer(1e6) 
batch_size = 128        # TODO tinker with this  

policy_net = PolicyNetwork(env)
policy_optim = torch.optim.Adam(policy_net.parameters(), lr)

q1_net = QNetwork(env)
q2_net = deepcopy(q1_net)     # Why not just call the constructor again?

q1_target = deepcopy(q1_net)
q2_target = deepcopy(q2_net)

q1_optim = torch.optim.Adam(q1_net.parameters(), lr)
q2_optim = torch.optim.Adam(q2_net.parameters(), lr)

def train():
    explore(10000)                              # Explore the environment by taking random actions 
    episode = 0                             
    while episode < max_episodes:
        s = env.reset()                         # Reset the environment w/o losing our transitions in the replay_buffer
        episodic_r =  0
        while True:
            with torch.no_grad():
                a = policy_net(s)               # action to be taken based on the state 
            s2, r, done, _ = env.step(2*a)      # why is this 2*a
            replay_buffer.store(s, a, r, s2, done)
            episodic_r += r 

            if done: 
                plot_reward(episode, episodic_r, '#4A04D4')
                episode += 1
                break
            else:
                s = s2 
            update(episode) 

def explore(steps):
    step = 0
    while step < steps:
        s = env.reset()     
        while True: 
            a = env.action_space.sample()
            s2, r, done, _ = env.step(a)
            replay_buffer.store(s, a, r, s2, done)
            step += 1
            if done: break
            else: s = s2 

def update(episode): 
    s, a, r, s2, done = replay_buffer.sample(batch_size)  
    a = a.squeeze().unsqueeze(1)
    with torch.no_grad():
        a2, π2 = policy_net.sample(s2)
        q1_next_max = q1_target(s2, a2)
        q2_next_max = q2_target(s2, a2)
        min_q = torch.min(q1_next_max, q2_next_max)

        y = r + done * gamma * (min_q - α*π2)      # TODO find this in the SAC paper

    q1_loss = F.mse_loss(q1_net(s, a), y)
    q2_loss = F.mse_loss(q2_net(s, a), y)
    plot_loss(episode, q1_loss, 'Q1', '#fe3b00')
    plot_loss(episode, q2_loss, 'Q2', '#004fff')


    # Template for optimization:
        # for input, target in dataset:
        #     optimizer.zero_grad()
        #     output = model(input)
        #     loss = loss_fn(output, target)
        #     loss.backward()
        #     optimizer.step()
    q1_optim.zero_grad()    # clears all optimizer torch.Tensors
    q1_loss.backward()     # compute the new gradients
    q1_optim.step()         # update the parameters accordingly

    q2_optim.zero_grad()    # clears all optimizer torch.Tensors
    q2_loss.backward()     # compute the new gradients
    q2_optim.step()         # update the parameters accordingly

    new_a, π = policy_net.sample(s)

    policy_loss = (α * π - q1_net(s, new_a)).mean()
    plot_loss(episode, policy_loss, 'π')

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # Update the q_target and policy_target
    for param, target_param in zip(q1_net.parameters(), q1_target.parameters()):
        target_param.data = target_param * tau + param.data*(1-tau)
    for param, target_param in zip(q2_net.parameters(), q2_target.parameters()):
        target_param.data = target_param * tau + param.data*(1-tau)

train()