import gym 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from model import * 
from replay_buffer import ReplayBuffer 
from visualize import * 
from copy import deepcopy
from random import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo_name = 'DDQN'       # Used for visualization 
max_episodes = 2000      # after 40 episodes, SAC flatlines around [-200, -100] reward
max_steps = 1000         # auto-terminate episode after win

gamma = 0.99            # Discount
α = 0.1                 # Entropy temperature -- relative importance vs. rewards
lr = 3e-4               # Determines how big of a gradient step to take when optimizing   
tau = 0.995             # Target smoothing coefficient --> how much of the old network(s) to keep
ε = 0.01                # Random exploration factor in training

env = gym.make('LunarLander-v2')
replay_buffer = ReplayBuffer(1e6) 
batch_size = 128      

q1 = Q(env)
q1_target = deepcopy(q1)
q1_optim = torch.optim.Adam(q1.parameters(), lr=lr)

q2 = Q(env)
q2_target = deepcopy(q2)
q2_optim = torch.optim.Adam(q2.parameters(), lr=lr)

def train():
    explore(10000)                                  # Explore the environment by taking random actions 
    episode = 0                                     
    while episode < max_episodes:                   # // rougly begin algorithm from SAC+ERE
        s = env.reset()                             # Get Initial state from environment 
        episodic_r =  0
        
        for step in range(max_steps):                  
            with torch.no_grad():
                if random() < ε:
                    a = env.action_space.sample()   # Take random action from available
                    env.render()
                else: 
                    a = int(np.argmax(q1(s)))                   # Sample action from policy
            
            s2, r, done, _ = env.step(a)          # Sample transition from environment //TODO why 2*a
            replay_buffer.store(s, a, r, s2, done)  # Add transition to the replay buffer
            episodic_r += r 

            if done: 
                plot_reward(episode, episodic_r, algo_name, '#4A04D4')
                episode += 1
                break
            else:
                s = s2 
            update(episode, step)                         # if s2 is a terminal state then -->update 


# explore to populate the replay buffer
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


def update(episode, step):
    """
    For each gradient step, adjust both Q networks, both target networks, as well as the policy network    
    """
    
    s, a, r, s2, done = replay_buffer.sample(batch_size)  
    
    with torch.no_grad():
        a2 = torch.argmax(q1_target(s2), dim=1, keepdim=True)           # Get action based on Q1
        #max_next_q = torch.gather(q2_target(s), -1, a2.long())         # Evaued by Q2
        max_next_q, _ = q2_target(s2).max(dim=1, keepdim=True)
        J = r + done*gamma*max_next_q 
    
    
    q1_loss = F.mse_loss(torch.gather(q1(s), 1, a.long()), J)
    q2_loss = F.mse_loss(torch.gather(q2(s), 1, a.long()), J)
    # Template for optimization:
        # for input, target in dataset:
        #     optimizer.zero_grad()
        #     output = model(input)
        #     loss = loss_fn(output, target)
        #     loss.backward()
        #     optimizer.step()

    q1_optim.zero_grad()
    q1_loss.backward()
    q1_optim.step() 

    q2_optim.zero_grad()
    q2_loss.backward()
    q2_optim.step()    
    
    # Update the q_target and policy_target
    for param, target_param in zip(q1.parameters(), q1_target.parameters()):
        target_param.data = target_param * tau + param.data*(1-tau)

    for param, target_param in zip(q2.parameters(), q2_target.parameters()):
        target_param.data = target_param * tau + param.data*(1-tau) 
    
    if step % 500 == 0: 
        plot_loss(episode, q1_loss, 'Q1', algo_name, color='#f44')
        #plot_loss(episode, q2_loss, 'Q2', algo_name, color='#FE3')
        # plot_loss(episode, q2_loss, 'Q2', algo_name, color='#F0F')


train()
