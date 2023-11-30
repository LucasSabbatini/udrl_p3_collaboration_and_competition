import numpy as np
from pprint import pprint
import torch
import os, sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import Agent

def test_agent(env, agent, brain_name):
    env_info = env.reset(train_mode = True)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    while True:
        actions, _, _= agent.act(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break
    return np.mean(scores)


def debug(item, name, print_e=False, only_shape=False):
    print("New item: {}".format(name))
    if only_shape:
        print(f"Shape: {item.shape}")
        return
    print(f"Type: {type(item)}")
    if print_e:
        pprint(item)
    try:
        print(f"Length: {len(item)}")
        pprint(f"First element: {item[0]}")
        pprint(f"Last element: {item[-1]}")
        try:
            print(f"Shape: {item.shape}")
        except:

            pass
        try:
            pprint("First element shape: {}".format(item[0].shape))
        except Exception as e:
            pprint(e)
    except:
        print("Object has no length")
    print("")
    
    
def load_trained_agent(env, ckpt_file):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]    
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    agent = Agent(num_agents, state_size, action_size)
    
    # policy_solution = torch.load('../ckpt/ppo_checkpoint_39.050999127142134.pth')
    policy_solution = torch.load(ckpt_file)
    agent.model.load_state_dict(policy_solution)
    # agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return agent


def save_training_scores(row_to_append, filename):
        
    with open(filename, 'a', newline='') as file:
        # Create a writer object
        writer = csv.writer(file)
        
        # Write the row to the file
        writer.writerow(row_to_append)