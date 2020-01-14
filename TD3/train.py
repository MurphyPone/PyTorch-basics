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

algo_name = 'TD3'
max_episodes = 2000
max_steps = 200
policy_delay = 2        # Update interval term


gamma = 0.99
lr = 3e-4
tau = 0.995

env = gym.make('Pendulum-v0')
replay_buffer = ReplayBuffer(1e6)
batch_size = 128

policy = PolicyGradient(env)
policy_target = deepcopy(policy)
policy_optim = torch.optim.Adam(policy.parameters(), lr=lr)

q1 = Q(env)
q1_target = deepcopy(q1)
q1_optim = torch.optim.Adam(q1.parameters(), lr=lr)

q2 = Q(env)
q2_target = deepcopy(q2)
q2_optim = torch.optim.Adam(q2.parameters(), lr=lr)


def train():
    explore(10000)
    episode = 0
    up_ct = 0                           # Update counter
    while episode < max_episodes:
        s = env.reset()         
        episodic_r = 0
        
        for step in range(max_steps):
            with torch.no_grad():
                a = policy(s) + add_noise()      # To boost out of local minima TODO make it epsilon greedy and add noise then
            s2, r, done, _ = env.step(a)
            replay_buffer.store(s, a, r, s2, done)
            episodic_r += r 

            if done:
                plot_reward(episode, episodic_r, algo_name, '#4A04D4')
                episode += 1 
                break 
            else: 
                s = s2
            update(episode, step, up_ct)
            up_ct += 1 


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

def update(episode, step, up_ct):
    """
    For each gradient step, ...  
    """

    s, a, r, s2, done = replay_buffer.sample(2*batch_size)  
    a = a.squeeze().unsqueeze(1)

    with torch.no_grad():
        max_next_a = policy_target(s2) + np.clip(add_noise(), -.4, -.4)
        q1_next_max = q1_target(s2, max_next_a)
        q2_next_max = q2_target(s2, max_next_a)
        min_q = torch.min(q1_next_max, q2_next_max)
        
        # Target for both loss functions
        J = r + done*gamma*min_q

    q1_loss = F.mse_loss(q1(s,a ), J)
    q2_loss = F.mse_loss(q2(s,a ), J)

    # Adjusting accordingly
    q1_optim.zero_grad()
    q1_loss.backward()
    q1_optim.step()

    q2_optim.zero_grad()
    q2_loss.backward()
    q2_optim.step()

    if step % 500 == 0: 
        plot_loss(episode, q1_loss, 'Q1', algo_name, color='#f44')
        plot_loss(episode, q2_loss, 'Q2', algo_name, color='#F95')

    update_policy(s, a, r, s2, done, episode, step, up_ct)


def update_policy(s, a, r, s2, done, episode, step, up_ct):
    if up_ct % policy_delay == 0:
        policy_loss = -(q1(s, policy(s))).mean()
        
        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()

        if step % 250 == 0: 
            plot_loss(episode, policy_loss, 'π', algo_name, color='#4A04D4')

        for param, target_param in zip(q1.parameters(), q1_target.parameters()):
            target_param.data = target_param.data*tau + param.data*(1-tau)
        for param, target_param in zip(q2.parameters(), q2_target.parameters()):
            target_param.data = target_param.data*tau + param.data*(1-tau)
        for param, target_param in zip(policy.parameters(), policy.parameters()):
            target_param.data = target_param.data*tau + param.data*(1-tau)


train()







