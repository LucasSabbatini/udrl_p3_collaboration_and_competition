import numpy as np
from collections import deque
import torch
from utils import test_agent
import os
from PPO import calculate_advantages
from agent import Agent
from unityagents import UnityEnvironment
import numpy as np
import time
import copy

def collect_trajectories(env, brain_name, agent, max_t):
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
        
    rollout = []
    agents_rewards = np.zeros(num_agents)
    episode_rewards = []

    for _ in range(max_t):
        actions, log_probs, values = agent.act(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards 
        dones = np.array([1 if t else 0 for t in env_info.local_done])
        agents_rewards += rewards

        for j, done in enumerate(dones):
            if dones[j]:
                episode_rewards.append(agents_rewards[j])
                agents_rewards[j] = 0

        rollout.append([states, values.detach(), actions.detach(), log_probs.detach(), rewards, 1 - dones])

        states = next_states

    pending_value = agent.model(states)[-1]
    returns = pending_value.detach() 
    rollout.append([states, pending_value, None, None, None, None])
    
    return rollout, returns, episode_rewards, np.mean(episode_rewards)


def train(env, brain_name, agent, num_agents, n_episodes, max_t, gamma=0.99, tau=0.95, run_name="testing_02", save_path=".."):
    print(f"Starting training...")
    time.sleep(2)
    env.info = env.reset(train_mode = True)[brain_name]
    all_scores = []
    all_scores_window = deque(maxlen=100)
    best_so_far = 0.0
        
    for i_episode in range(n_episodes):
        # Each iteration, N parallel actors collect T time steps of data
        rollout, returns, _, _ = collect_trajectories(env, brain_name, agent, max_t)
        
        states, actions, log_probs_old, returns, advantages = calculate_advantages(rollout, returns, num_agents, gamma=gamma, tau=tau)
        # print(f"States: {states.shape}. Actions: {actions.shape}. Log_probs_old: {log_probs_old.shape}. Returns: {returns.shape}. Advantages: {advantages.shape}")
        agent.learn(states, actions, log_probs_old, returns, advantages)
        
        test_mean_reward = test_agent(env, agent, brain_name)

        all_scores.append(test_mean_reward)
        all_scores_window.append(test_mean_reward)

        # if np.mean(all_scores_window) > best_so_far:
        #     if not os.path.isdir(f"{save_path}/ckpt/{run_name}/"):
        #         os.mkdir(f"{save_path}/ckpt/{run_name}/")
        #     torch.save(agent.model.state_dict(), f"{save_path}/ckpt/{run_name}/ppo_checkpoint_{np.mean(all_scores_window)}.ckpt")
        #     best_so_far = np.mean(all_scores_window)
        #     if np.mean(all_scores_window) > 30:
                
        #         print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(all_scores_window)))
        #         # break       
        
        if (i_episode + 1) % 20 == 0:
            
            if np.mean(all_scores_window) > best_so_far:
                if not os.path.isdir(f"{save_path}/ckpt/{run_name}/"):
                    os.mkdir(f"{save_path}/ckpt/{run_name}/")
                torch.save(agent.model.state_dict(), f"{save_path}/ckpt/{run_name}/ep_{i_episode}_avg_score_{np.mean(all_scores_window)}.ckpt")
                best_so_far = np.mean(all_scores_window)
                print(f"Saved checkpoint with average score {np.mean(all_scores_window)}")
                if np.mean(all_scores_window) > 30:        
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(all_scores_window)))
                    # break       
            
            print('Episode {}, Total score this episode: {}, Last {} average: {}'.format(i_episode + 1, test_mean_reward, min(i_episode + 1, 100), np.mean(all_scores_window)) )
        
    save_scores(all_scores, run_name, save_path)
    return all_scores

def save_scores(scores, run_name, save_path):
    if not os.path.isdir(f"{save_path}/{run_name}/"):
        os.mkdir(f"{save_path}/{run_name}/")
    np.save(f"{save_path}/{run_name}/scores.npy", scores)


def train_run(parms):
    """Trains a single agent with the given hyperparameters"""
    
    # Training Hyperparameters
    EPISODES = 10000        # Number of episodes to train for
    # MAX_T = 2048          # Max length of trajectory
    MAX_T = 1000            # Max length of trajectory
    SGD_EPOCHS = parms["sgd_epochs"]          # Number of gradient descent steps per batch of experiences
    BATCH_SIZE = parms["batch_size"]         # minibatch size
    BETA = 0.01             # entropy regularization parameter
    GRADIENT_CLIP = parms["gradient_clip"]       # gradient clipping parameter

    # optimizer parameters
    # LR = 5e-4               # learning rate
    LR = parms["lr"]               # learning rate
    OP_EPSILON = 1e-5       # optimizer epsilon
    WEIGHT_DECAY = parms["weight_decay"]    # L2 weight decay

    # PPO parameters
    GAMMA = 0.99            # Discount factor
    TAU = parms["tau"]              # GAE parameter
    # PPO_CLIP_EPSILON = 0.1  # ppo clip parameter
    PPO_CLIP_EPSILON = parms["ppo_clip_epsilon"]  # ppo clip parameter
    STD = parms["std"]

    env = UnityEnvironment(file_name="../../unity_ml_envs/Tennis_Windows_x86_64/Tennis.exe")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    time.sleep(2)

    # Environment variables
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    print(f"Number of agents: {num_agents}. State size: {state_size}. Action size: {action_size}")
    # Instantiate the agent
    agent = Agent(num_agents, state_size, action_size,
                  LR=LR,
                  op_epsilon=OP_EPSILON,
                  weight_decay=WEIGHT_DECAY,
                  batch_size=BATCH_SIZE,
                  sgd_epochs=SGD_EPOCHS,
                  gradient_clip=GRADIENT_CLIP,
                  std=STD,
                  value_size=1,
                  hidden_size=64,
                  clip_epsilon=PPO_CLIP_EPSILON,
                  c1=0.5,
                  beta=BETA)
    train(env, brain_name, agent, num_agents, EPISODES, MAX_T,
          gamma=GAMMA, tau=TAU, run_name=parms["run_name"], save_path=parms["save_path"])


def train_multiple(params_list):
    """Trains multiple agents with different hyperparameters"""
    
    sgd_epochs = [3, 4, 5]
    batch_sizes = [32]
    gradient_clips = [1, 3, 5]
    ppo_clip_epsilons = [0.1, 0.2, 0.4]
    lrs = [1e-4, 3e-4]
    weight_decays = [1e-4, 3e-4]
    gae_tau = [0.95, 0.99]
    stds = [0.0, 0.1, 0.2]
    
    params_list = []
    for sgd_epoch in sgd_epochs:
        for batch_size in batch_sizes:
            for gradient_clip in gradient_clips:
                for ppo_clip_epsilon in ppo_clip_epsilons:
                    for lr in lrs:
                        for weight_decay in weight_decays:
                            for tau in gae_tau:
                                for std in stds:
                                    params_list.append({"std": copy(std), 
                                                        "tau": copy(tau),
                                                        "sgd_epochs": copy(sgd_epoch),
                                                        "batch_size": copy(batch_size),
                                                        "gradient_clip": copy(gradient_clip),
                                                        "ppo_clip_epsilon": copy(ppo_clip_epsilon),
                                                        "lr": copy(lr),
                                                        "weight_decay": copy(weight_decay),
                                                        "run_name": f"sgd_epochs_{sgd_epoch}_batch_size_{batch_size}_gradient_clip_{gradient_clip}_ppo_clip_epsilon_{ppo_clip_epsilon}_lr_{lr}_weight_decay_{weight_decay}",
                                                        "save_path": ".."})
    
    for params in params_list:
        print(f"Starting training with parameters {params['run_name']}")
        train_run(params)
        


if __name__=="__main__":

    
    # Training Hyperparameters
    EPISODES = 10000        # Number of episodes to train for
    # MAX_T = 2048          # Max length of trajectory
    MAX_T = 1000            # Max length of trajectory
    SGD_EPOCHS = 4          # Number of gradient descent steps per batch of experiences
    BATCH_SIZE = 32         # minibatch size
    BETA = 0.01             # entropy regularization parameter
    GRADIENT_CLIP = 5       # gradient clipping parameter

    # optimizer parameters
    # LR = 5e-4               # learning rate
    LR = 3e-4               # learning rate
    OP_EPSILON = 1e-5       # optimizer epsilon
    WEIGHT_DECAY = 1.E-4    # L2 weight decay

    # PPO parameters
    GAMMA = 0.99            # Discount factor
    TAU = 0.95              # GAE parameter
    # PPO_CLIP_EPSILON = 0.1  # ppo clip parameter
    PPO_CLIP_EPSILON = 0.2  # ppo clip parameter

    env = UnityEnvironment(file_name="../../unity_ml_envs/Tennis_Windows_x86_64/Tennis.exe")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    time.sleep(2)

    # Environment variables
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    print(f"Number of agents: {num_agents}. State size: {state_size}. Action size: {action_size}")
    # Instantiate the agent
    agent = Agent(num_agents, state_size, action_size,
                  LR=LR,
                  op_epsilon=OP_EPSILON,
                  weight_decay=WEIGHT_DECAY,
                  batch_size=BATCH_SIZE,
                  sgd_epochs=SGD_EPOCHS,
                  gradient_clip=GRADIENT_CLIP,
                  std=0.0,
                  value_size=1,
                  hidden_size=64,
                  clip_epsilon=PPO_CLIP_EPSILON,
                  c1=0.5,
                  beta=BETA)

    # Train the agent
    print(f"Starting training with parameters LR={LR}, WEIGHT_DECAY={WEIGHT_DECAY}, BATCH_SIZE={BATCH_SIZE}, SGD_EPOCHS={SGD_EPOCHS}, GRADIENT_CLIP={GRADIENT_CLIP}, BETA={BETA}, GAMMA={GAMMA}, TAU={TAU}, PPO_CLIP_EPSILON={PPO_CLIP_EPSILON}")
    # exit()
    train(env, brain_name, agent, num_agents, EPISODES, MAX_T,
          gamma=GAMMA, tau=TAU, run_name="testing_02", save_path="..")
    env.close()