import numpy as np
from pprint import pprint
import torch
import os, sys
import csv
import matplotlib.pyplot as plt

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
        
        
# CSV file format 
def save_scores_csv(scores, run_name, save_path):
    # Ensure the directory exists
    full_path = f"{save_path}/ckpt/{run_name}"
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    # File path for the CSV file
    file_path = f"{full_path}/scores.csv"

    # Write scores to a CSV file
    with open(file_path, 'w', newline='') as csvfile:
        score_writer = csv.writer(csvfile)
        score_writer.writerow(['Episode', 'Score'])  # Header
        for i, score in enumerate(scores, 1):  # Enumerate starts at 1 for episode numbering
            score_writer.writerow([i, score])
    
    print(f"Scores saved to {file_path}")

def load_scores_from_csv(csv_file_path):
    scores = []

    # Read scores from CSV
    with open(csv_file_path, 'r') as csvfile:
        score_reader = csv.reader(csvfile)
        next(score_reader, None)  # Skip the header
        for row in score_reader:
            scores.append(float(row[1]))  # Assuming score is in the second column

    return scores

def read_and_plot_scores_csv(csv_file_path):
    scores = []

    # Read scores from CSV
    with open(csv_file_path, 'r') as csvfile:
        score_reader = csv.reader(csvfile)
        next(score_reader, None)  # Skip the header
        for row in score_reader:
            scores.append(float(row[1]))  # Assuming score is in the second column

    # Plotting
    plt.plot(scores)
    plt.title('Scores Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()
    

# NPY file format
def save_scores_npy(scores, run_name, save_path):
    print(f"Saving scores to {save_path}/ckpt/{run_name}/scores.npy")
    if not os.path.isdir(f"{save_path}/ckpt/{run_name}/"):
        os.mkdir(f"{save_path}/ckpt/{run_name}")
    np.save(f"{save_path}/ckpt/{run_name}/scores.npy", scores)

def load_scores_npy(run_name, save_path):
    file_path = f"{save_path}/ckpt/{run_name}/scores.npy"

    if not os.path.isfile(file_path):
        print(f"No saved scores found at {file_path}")
        return None

    scores = np.load(file_path)
    print(f"Scores loaded from {file_path}")
    return scores

def read_and_plot_scores_npy(run_name, save_path):
    file_path = f"{save_path}/ckpt/{run_name}/scores.npy"

    if not os.path.isfile(file_path):
        print(f"No saved scores found at {file_path}")
        return

    scores = np.load(file_path)

    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Scores per Episode')
    plt.title('Training Progress')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.grid(True)
    plt.legend()
    plt.show()

    
    
    


