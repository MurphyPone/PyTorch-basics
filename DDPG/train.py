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

algo_name = 'DDPG'
max_episodes = 2000
max_steps = 200         

gamma = 0.99
lr = 3e-4
tau = 0.995

env = gym.make('Pendulum-v0')
replay_buffer = ReplayBuffer(1e6)
batch_size = 128

policy = PolicyGradient(env)
policy_target = deepcopy(policy)
policy_optim = torch.optim.Adam(policy.parameters(), lr=lr)

q = Q(env)
q_target = deepcopy(q)
q_optim = torch.optim.Adam(q.parameters(), lr=lr)


def train():
    explore(10000)
    episode = 0
    while episode < max_episodes:
        s = env.reset()         
        episodic_r = 0
        
        for step in range(max_steps):
            with torch.no_grad():
                a = policy(s) + add_noise()      # TODO Why is it necessary to add noise?
            s2, r, done, _ = env.step(a)
            replay_buffer.store(s, a, r, s2, done)
            episodic_r += r 

            if done:
                plot_reward(episode, episodic_r, algo_name, '#4A04D4')
                episode += 1 
                break 
            else: 
                s = s2
            update(episode, step)


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


def add_noise():
    return np.clip(np.random.normal(0,.15), -.3,.3)

def update(episode, step):
    """
    For each gradient step, ...  
    """

    s, a, r, s2, done = replay_buffer.sample(2*batch_size)  
    a = a.squeeze().unsqueeze(1)

    with torch.no_grad():
        max_next_a = policy_target(s2)
        J = r + done*gamma*q_target(s2, max_next_a)

    # Calculate loss and Update q and policy
    q_loss = F.mse_loss(q(s, a), J)

    q_optim.zero_grad()
    q_loss.backward()
    q_optim.step()

    policy_loss = -(q(s, policy(s))).mean()
    
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    if step % 500 == 0: 
        plot_loss(episode, policy_loss, 'π', algo_name, color='#f44')
        plot_loss(episode, q_loss, 'Q', algo_name, color='#F95')

    # Update q_target and policy_target
    for param, target_param in zip(q.parameters(), q_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)
    for param, target_param in zip(policy.parameters(), policy_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)

train()







