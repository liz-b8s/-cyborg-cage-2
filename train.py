# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
import sys

# You may need to add the path to CybORG if you encounter errors such as 'Can't import CybORG'
sys.path.append('/Cage_2_RS/cage_challenge_2/CybORG')

import numpy as np
from tqdm import tqdm
import os
import glob
import gym as gym
import pandas as pd
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect
from Agents.PPOAgent import PPOAgent
import random
import torch
import argparse
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'


def train(env, experiment, iteration, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, best_ckpt_folder, log_dir, print_interval=10, save_interval=100,
          start_actions=[]):
    agent = PPOAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions)

    running_reward, time_step = 0, 0
    alt_running_rew = 0
    model_rewards_adjusted = []
    model_rewards_real = []
    best_reward = -1000

    writer = SummaryWriter(log_dir=log_dir)

    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            action = agent.get_action(state)
            state, reward_adjusted, reward_real, done, _ = env.step(action)
            # print(reward)

            agent.store(reward_adjusted, done)

            if time_step % update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0

            alt_running_rew += reward_adjusted

            # Keep this as the 'real' reward, so it's able to be plotted against baselines
            running_reward += reward_real

        agent.end_episode()

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            print('Checkpoint saved')

        if i_episode % print_interval == 0:
            running_reward = int((running_reward / print_interval))
            alt_running_rew = int((alt_running_rew / print_interval))

            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
            writer.add_scalar('training Running Reward Real', running_reward, i_episode)
            writer.add_scalar('training Running Reward Augmented', alt_running_rew, i_episode)

            if running_reward > best_reward:
                best_reward = running_reward
                ckpt = os.path.join(best_ckpt_folder, 'best_model.pth')
                torch.save(agent.policy.state_dict(), ckpt)
                print('Best Checkpoint saved')

            model_rewards_real.append(running_reward)
            model_rewards_adjusted.append(alt_running_rew)

    model_rewards_folder = os.path.join(os.getcwd(), "Model_rewards", 'Reward_shaping_vm', str(experiment))
    if not os.path.exists(model_rewards_folder):
        os.makedirs(model_rewards_folder)

    specific_model_real_dir = 'Reward_shaping_real_iter_' + str(iteration)
    pd.DataFrame(model_rewards_real).to_csv(model_rewards_folder + '/' + specific_model_real_dir, index=False, sep=',')

    specific_model_adj_dir = 'Reward_shaping_adj_iter_' + str(iteration)
    pd.DataFrame(model_rewards_adjusted).to_csv(model_rewards_folder + '/' + specific_model_adj_dir, index=False,
                                                sep=',')


################# Training #################
# This is for labeling the model rewards csv files, this folder will contain all the rewards for each model.
experiment_name = 'log_exp'
# RS_baseline_lm_100k_rand_1
# How many models will be run: (30 would be ideal)
max_model_runs = 1
max_training_episodes = 2500
SEED = 1


# if __name__ == '__main__':

def run_script(experiment_name, max_model_runs, max_training_episodes, SEED):
    for trial in tqdm(range(max_model_runs)):

        iteration = trial

        # set seeds for reproducibility
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        # change checkpoint directory
        # folder = 'bline'
        specific_model_folder = "Reward_shaping_vm/" + str(experiment_name) + '_' + str(iteration)
        specific_best_model_folder = specific_model_folder + "/best_model/" + str(experiment_name) + '_' + str(
            iteration)
        print(os.getcwd())
        ckpt_folder = os.path.join(os.getcwd(), "Models", specific_model_folder)
        log_folder = os.path.join(os.getcwd(), "log", specific_model_folder)
        best_ckpt_folder = os.path.join(os.getcwd(), "Models", specific_best_model_folder)

        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)  # on tensorboard:  log/Reward_shaping_test
        if not os.path.exists(best_ckpt_folder):
            os.makedirs(best_ckpt_folder)

        # writer = SummaryWriter(log_dir=log_folder)

        CYBORG = CybORG(PATH, 'sim', agents={
            'Red': RedMeanderAgent
        })
        env = ChallengeWrapper2(env=CYBORG, agent_name="Blue")
        input_dims = env.observation_space.shape[0]

        action_space = [133, 134, 135, 139]  # restore enterprise and opserver
        action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
        action_space += [16, 17, 18, 22]  # remove enterprise and opserer
        action_space += [11, 12, 13, 14]  # analyse user hosts
        action_space += [141, 142, 143, 144]  # restore user hosts
        action_space += [132]  # restore defender
        action_space += [2]  # analyse defender
        action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts

        start_actions = [1004, 1004, 1000]  # user 2 decoy * 2, ent0 decoy

        print_interval = 10
        save_interval = 1000
        max_episodes = max_training_episodes
        max_timesteps = 100
        # 200 episodes for buffer
        update_timesteps = 20000
        K_epochs = 6
        eps_clip = 0.2
        gamma = 0.99
        lr = 0.002
        experiment = experiment_name

        train(env, experiment, iteration, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=[0.9, 0.990], ckpt_folder=ckpt_folder, best_ckpt_folder=best_ckpt_folder, log_dir=log_folder,
              print_interval=print_interval, save_interval=save_interval, start_actions=start_actions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed value')
    parser.add_argument('--output', type=str, help='Output file name')
    parser.add_argument('--max_model_runs', type=int, help='Maximum number of model runs')
    parser.add_argument('--max_training_episodes', type=int, help='Maximum number of training episodes')
    args = parser.parse_args()

    run_script(args.output, args.max_model_runs, args.max_training_episodes, args.seed)
