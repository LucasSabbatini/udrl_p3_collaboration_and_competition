import torch

# Loss function. NOT INTEGRATED YET
def ppo_loss(model, states, actions, log_probs_old, returns, advantages, clip_epsilon=0.1, c1=0.5, beta=0.01):
    dist, values = model(states)
    
    log_probs = dist.log_prob(actions)
    log_probs = torch.sum(log_probs, dim=1, keepdim=True)
    entropy = dist.entropy().mean()
    
    # r(θ) =  π(a|s) / π_old(a|s)
    ratio = (log_probs - log_probs_old).exp() # NOTE WHYYYYYYY????
    
    # Surrogate Objctive : L_CPI(θ) = r(θ) * A
    obj = ratio * advantages 
    
    # clip ( r(θ), 1-Ɛ, 1+Ɛ )*A
    obj_clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    
    # L_CLIP(θ) = E { min[ r(θ)A, clip ( r(θ), 1-Ɛ, 1+Ɛ )*A ] - β * KL }
    policy_loss = -torch.min(obj, obj_clipped).mean(0) - beta * entropy.mean() # NOTE: WHY ARE WE TAKING THE MEAN AGAIN???
    
    # L_VF(θ) = ( V(s) - V_t )^2
    value_loss = c1 * (returns - values).pow(2).mean()
    
    return policy_loss + value_loss


def calculate_advantages(rollout, returns, num_agents, gamma=0.99, tau=0.95):
    """ Given a rollout, calculates the advantages for each state """
    num_steps = len(rollout) - 1
    processed_rollout = [None] * num_steps
    advantages = torch.zeros((num_agents, 1))

    for i in reversed(range(num_steps)):
        states, value, actions, log_probs, rewards, dones = map(lambda x: torch.Tensor(x), rollout[i])
        next_value = rollout[i + 1][1]

        dones = dones.unsqueeze(1)
        rewards = rewards.unsqueeze(1)

        # Compute the updated returns
        returns = rewards + gamma * dones * returns

        # Compute temporal difference error
        td_error = rewards + gamma * dones * next_value.detach() - value.detach()
        
        advantages = advantages * tau * gamma * dones + td_error
        processed_rollout[i] = [states, actions, log_probs, returns, advantages]

    # Concatenate along the appropriate dimension
    states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return states, actions, log_probs_old, returns, advantages