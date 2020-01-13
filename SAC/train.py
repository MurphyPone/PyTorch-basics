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

algo_name = 'SAC'       # Used for visualization 
max_episodes = 2000     # after 40 episodes, SAC flatlines around [-200, -100] reward
max_steps = 200         # auto-terminate episode after > 200 steps 

gamma = 0.99            # Discount
α = 0.1                 # Entropy temperature -- relative importance vs. rewards
lr = 3e-4               # Determines how big of a gradient step to take when optimizing   
tau = 0.995             # Target smoothing coefficient --> how much of the old network(s) to keep

env = gym.make('Pendulum-v0')
replay_buffer = ReplayBuffer(1e6) 
batch_size = 128        

policy_net = PolicyNetwork(env)
policy_optim = torch.optim.Adam(policy_net.parameters(), lr)

q1_net = QNetwork(env)
q2_net = QNetwork(env)               # //TODO 

q1_target = deepcopy(q1_net)
q2_target = deepcopy(q2_net)

q1_optim = torch.optim.Adam(q1_net.parameters(), lr)
q2_optim = torch.optim.Adam(q2_net.parameters(), lr)

def train():
    explore(10000)                                  # Explore the environment by taking random actions 
    episode = 0                                     
    while episode < max_episodes:                   # // rougly begin algorithm from SAC+ERE
        s = env.reset()                             # Get Initial state from environment 
        episodic_r =  0
        
        for step in range(max_steps):                  
            with torch.no_grad():
                a = policy_net(s)                   # Sample action from policy
            s2, r, done, _ = env.step(2*a)          # Sample transition from environment //TODO why 2*a
            replay_buffer.store(s, a, r, s2, done)  # Add transition to the replay buffer
            episodic_r += r 

            if done: 
                plot_reward(episode, episodic_r, '#4A04D4')
                episode += 1
                break
            else:
                s = s2 
            update(step)                         # if s2 is a terminal state then -->update 


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


def update(episode):
    """
    For each gradient step, adjust both Q networks, both target networks, as well as the policy network    
    """
    
    s, a, r, s2, done = replay_buffer.sample(batch_size)  
    a = a.squeeze().unsqueeze(1)
    
    with torch.no_grad():
        a2, π2 = policy_net.sample(s2)
        q1_next_max = q1_target(s2, a2)     # TODO Pretty sure this is the vanilla implementation of KL-Divergence part?
        q2_next_max = q2_target(s2, a2)     # Use targets to slow down/stabilize
        min_q = torch.min(q1_next_max, q2_next_max) # Take the min to balance over-reward for seeing a new state

        J = r + done*gamma*(min_q - α*π2)      # difference between the min Q target and the entropy sampled policy

    # Calc loss based on the current marginals and the evaled objective using s2
    q1_loss = F.mse_loss(q1_net(s, a), J)
    q2_loss = F.mse_loss(q2_net(s, a), J)

    # Template for optimization:
        # for input, target in dataset:
        #     optimizer.zero_grad()
        #     output = model(input)
        #     loss = loss_fn(output, target)
        #     loss.backward()
        #     optimizer.step()
    
    q1_optim.zero_grad()    # clears all optimizer torch.Tensors
    q1_loss.backward()      # compute the new gradients
    q1_optim.step()         # update the parameters accordingly

    q2_optim.zero_grad()    # clears all optimizer torch.Tensors
    q2_loss.backward()      # compute the new gradients
    q2_optim.step()         # update the parameters accordingly

    a2, π2 = policy_net.sample(s)

    policy_loss = (α*π2 - q1_net(s, a2)).mean()   # use one network for consistency

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # Update the q_target and policy_target
    for param, target_param in zip(q1_net.parameters(), q1_target.parameters()):
        target_param.data = target_param * tau + param.data*(1-tau)
    for param, target_param in zip(q2_net.parameters(), q2_target.parameters()):
        target_param.data = target_param * tau + param.data*(1-tau)

    # if episode % 5 == 0: 
    #     plot_loss(episode, policy_loss, 'π', '#000000')
    #     plot_loss(episode, q1_loss, 'Q1', '#fe3b00')
    #     plot_loss(episode, q2_loss, 'Q2', '#004fff')

train()
