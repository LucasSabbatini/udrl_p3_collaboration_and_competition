

# Agent and models
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Agent hyperparameters
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # Discount factor
TAU = 0.95              # GAE parameter
BETA = 0.01             # entropy regularization parameter
PPO_CLIP_EPSILON = 0.2  # ppo clip parameter
GRADIENT_CLIP = 5       # gradient clipping parameter


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x)) # still the same, since the range is [-1.0, +1.0]
    
    
class Critic(nn.Module):
    def __init__(self, state_size, value_size=1, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, value_size)
        
    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, value_size=1, hidden_size=64, std=0.0):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, value_size, hidden_size)
        
        self.log_std = nn.Parameter(torch.ones(1, action_size)*std)
        
    def forward(self, states): # TODO: LEARN WHAT THE FUCK THIS DOES
        obs = torch.FloatTensor(states)
        
        # Critic
        values = self.critic(obs)
        
        # Actor
        mu = self.actor(obs)
        std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        
        return dist, values
    

class Agent():
    def __init__(self, num_agents, state_size, action_size):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.model = ActorCritic(state_size, action_size, value_size=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=EPSILON)
        self.model.train()
        
    def act(self, states): # TODO: IS THIS CORRECT? WE SHOULD USE MU AS THE ACTIONS, AND NOT SAMPLE FROM THE DISTRIBUTION ???
        """Remember: states are state vectors for each agent
        It is used when collecting trajectories
        """
        dist, values = self.model(states) # pass the state trough the network and get a distribution over actions and the value of the state
        actions = dist.sample() # sample an action from the distribution
        log_probs = dist.log_prob(actions) # calculate the log probability of that action
        log_probs = log_probs.sum(-1).unsqueeze(-1) # sum the log probabilities of all actions taken (in case of multiple actions) and reshape to (batch_size, 1)
        
        return actions, log_probs, values
    
    def learn(self, states, actions, log_probs_old, returns, advantages, sgd_epochs=4):
        """ Performs a learning step given a batch of experiences
        
        Remmeber: in the PPO algorithm, we perform SGD_episodes (usually 4) weights update steps per batch
        using the proximal policy ratio clipped objective function
        """        

        num_batches = states.size(0) // BATCH_SIZE
        for i in range(sgd_epochs):
            batch_count = 0
            batch_ind = 0
            for i in range(num_batches):
                sampled_states = states[batch_ind:batch_ind+BATCH_SIZE, :]
                sampled_actions = actions[batch_ind:batch_ind+BATCH_SIZE, :]
                sampled_log_probs_old = log_probs_old[batch_ind:batch_ind+BATCH_SIZE, :]
                sampled_returns = returns[batch_ind:batch_ind+BATCH_SIZE, :]
                sampled_advantages = advantages[batch_ind:batch_ind+BATCH_SIZE, :]
                
                L = ppo_loss(self.model, sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages)
                
                self.optimizer.zero_grad()
                (L).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
                self.optimizer.step()
                
                batch_ind += BATCH_SIZE
                batch_count += 1

