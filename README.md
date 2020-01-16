# PyTorch-basics
A series of algorithms, and exercises made in an effort to learn PyTorch.  

# Requirements 
`pip install torch torchvision visdom`

# Use 
1. `visdom`
2. Navigate to any of the algorithm directories and execute `python train.py`

# Contents 
- **A2C**: Advantage Actor-Critic is an on-policy algorithm which maximizes performance via gradient ascent while also accounting for an Advantage value (A = Q - V, e.g. how much better is this action compared to the average action taken in this state) which is approximated by a target value network.  
- **DQN**: Deep Q-Learning is an off-policy algorithm which learns which actions maximize the attempts optimal Q function for a given environment.
- **DDQN**: Double Deep Q-Learning corrects some of the bias flaws of a simple DQN.  By introducing a second Q-Network to evaluate state-action pairs, the model is less likely to excessively reward new states.  One network selects the action, and the other network evaluates the efficacy of that selection.  
- **DDPG**: Deep Deterministic Policy Gradient is an off-policy algorithm which simultanesouly learns a Q-function as well as a deterministics policy π whcih allows it to act in continuous actions spaces.  The Q-function is improved via the Bellman equation, and, in turn, the policy learns from the Q-function.  
- **SAC**: Soft Actor-Critic improves on DDPG and other algorithms by modifying the objective function from simply cumulative reward over state-action pairs to include an entropy term which rewards stochastic policy selection.  The algorithm uses processes similar to DDQN to optimize the reward as well as the entropy term. 
    - **SAC-ERE**: Soft Actor-Critic with Emphasizing Recent Experience.  This implementation employs a more-sophisticated replay buffer sampling method than the uniform sampling initially proposed by Haarnoja et al.  By annealing the replay buffer batch size over as an episode progresses, the model is less likely to sample irrelevant data.  ERE accelerates the marginal-discard process while still guaranteeing high-entropy sampling via vanilla SAC.  

# References  
- https://github.com/BCHoagland/ChaRLes
- https://github.com/rileyp2000/MLPytorchWork
- [OpenAI](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- https://murphypone.github.io/blog/ml-pm-1

# TODO 
- Flesh out documentation across implementations or at least refer to SAC-ERE
- Standardize visualization
- Implement ε-greedy exploration in all cases
- Implmenent GAN
