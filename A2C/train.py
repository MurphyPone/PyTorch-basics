import gym 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from model import * 
from replay_buffer import ReplayBuffer 
from visualize import * 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

algo_name = 'A2C'
max_episodes = 2000
max_steps = 200         # CartPole terminates after 200 steps, solved when avg reward ≥ 195 for ≥ 100 consecutive trials  

gamma = 0.99
lr = 3e-4

env = gym.make('CartPole-v1')
replay_buffer = ReplayBuffer(1e6)
batch_size = 128

actor = Actor(env)
actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)

critic = Critic(env)
critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)

def train():
    explore(10000)
    episode = 0
    while episode < max_episodes:
        s = env.reset()         
        episodic_r = 0
        
        for step in range(max_steps):
            with torch.no_grad():
                a = actor(s)
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

def update(episode, step):
    """
    For each gradient step, ... TODO 
    """
    # TODO I'm updating in batches... not necessarily corrrect s
    s, a, r, s2, done = replay_buffer.sample(2*batch_size)  
    a = a.squeeze().unsqueeze(1)

    # Calculate returns 
    returns = [0] * len(r)
    discounted_next = 0

    for i in reversed(range(len(r))):
        returns[i] = r[i] + discounted_next    
        discounted_next = gamma * returns[i] * done[i-1] # TODO for some reason this makes it worse
    returns = torch.stack(returns)

    with torch.no_grad():
        # Calculate and normalize advantage
        adv = returns - critic(s)
        mean = adv.mean()
        std = adv.std()
        adv = (adv - mean) / (std + 1e-6)   # Add 1e-6 in case all rewards are the same to prevent division by std=0           

    # Calculate the log_probabilities because: https://stats.stackexchange.com/questions/87182/what-is-the-role-of-the-logarithm-in-shannons-entropy
    log_p = actor.get_log_prob(s, a)

    # Calculate loss and update actor/critic accordingly 
    actor_loss = -(adv * log_p).mean()
    
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # critic_loss = F.mse_loss(returns, critic(s)) 
    critic_loss = ((returns - critic(s)) ** 2).mean()
    
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()


    if step % 500 == 0: 
        plot_loss(episode, actor_loss, 'π', algo_name, color='#f44')
        plot_loss(episode, critic_loss, 'V', algo_name, color='#F95')

train()







