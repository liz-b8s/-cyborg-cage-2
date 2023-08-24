import sys
# You may need to add the path to CybORG if you encounter errors such as 'Can't import CybORG'
sys.path.append('/Cage_2_RS/cage_challenge_2/CybORG')

import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
import random


MAX_EPS = 1000
agent_name = 'Blue'
random.seed(0)


def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')


disprop_scale = []
norm_mag = []
scaleup_mag = []
baseline = []
small_pos_b = []
small_pos_m = []
big_pos_m = []
big_pos_b = []
icm_b = []
icm_m = []
b_baseline = []
m_baseline = []
b_baseline_ext = []

# Adjust by how many models you ran. In the paper, we ran 15 for each experiment.
no_models = 15


for i in range(0, no_models, 1):
    i_new = str(i+1)
    path_b_baseline = '/Models/bline/Reward_shaping_vm/RS_baseline_vm__' + i_new + '_0/best_model/RS_baseline_vm__' + i_new + '_0/best_model.pth'
    path_m_baseline = '/Models/bline/Reward_shaping_vm/RS_baseline_meander_vm_' + i_new + '_0/best_model/RS_baseline_meander_vm_' + i_new + '_0/best_model.pth'

    # Exp1
    # Below paths are edited depending on whether meander or bline models are being evaluated
    path_disprop = '/Models/bline/Reward_shaping_vm/RS_disprop_vm_meander_' + i_new +'_0/best_model/RS_disprop_vm_meander_' + i_new +'_0/best_model.pth'
    path_norm = '/Models/bline/Reward_shaping_vm/RS_norm_vm_meander_' + i_new +'_0/best_model/RS_norm_vm_meander_' + i_new +'_0/best_model.pth'
    path_scaleup = '/Models/bline/Reward_shaping_vm/RS_scaleup_vm_meander_' + i_new +'_0/best_model/RS_scaleup_vm_meander_' + i_new +'_0/best_model.pth'

    # Exp2
    path_small_pos_b =  '/Models/bline/Reward_shaping_vm/RS_small_pos_vm_' + i_new +'_0/best_model/RS_small_pos_vm_' + i_new +'_0/best_model.pth'
    path_small_pos_m = '/Models/bline/Reward_shaping_vm/RS_small_pos_vm_meander_' + i_new +'_0/best_model/RS_small_pos_vm_meander_' + i_new +'_0/best_model.pth'

    path_big_pos_b = '/Models/bline/Reward_shaping_vm/RS_big_pos_vm_' + i_new +'_0/best_model/RS_big_pos_vm_' + i_new +'_0/best_model.pth'
    path_big_pos_m = '/Models/bline/Reward_shaping_vm/RS_big_pos_meander_vm_' + i_new + '_0/best_model/RS_big_pos_meander_vm_' + i_new + '_0/best_model.pth'

    # Exp3
    path_icm_b = '/Models/bline/Reward_shaping_vm/RS_ICM_' + i_new + '_0/best_model/RS_ICM_' + i_new + '_0/best_model.pth'
    path_icm_m = '/Models/bline/Reward_shaping_vm/RS_ICM_Meander_' + i_new + '_0/best_model/RS_ICM_Meander_' + i_new + '_0/best_model.pth'

    # Extension
    path_icm_b_ext = '/Models/Reward_shaping_vm/ICM_eta_i_0' + i_new + '_0/best_model/ICM_eta_i_0' + i_new + '_0/best_model.pth'
    path_icm_m_ext = '/Models/Reward_shaping_vm/ICM_eta_i_0' + i_new + '_0/best_model/ICM_eta_i_0' + i_new + '_0/best_model.pth'


    # baseline appending
    b_baseline.append(path_b_baseline)
    m_baseline.append(path_m_baseline)

    # Exp1 appending
    disprop_scale.append(path_disprop)
    scaleup_mag.append(path_scaleup)
    norm_mag.append(path_norm)

    # Exp2 appending
    small_pos_b.append(path_small_pos_b)
    small_pos_m.append(path_small_pos_m)

    big_pos_b.append(path_big_pos_b)
    big_pos_m.append(path_big_pos_m)

    # Exp3 appending
    icm_b.append(path_icm_b)
    icm_m.append(path_icm_m)

    # Extension appending
    b_baseline_ext.append(path_icm_b_ext)

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'

    # If you're evaluating bline agents, make bline True.
    # If you're evaluating Meander agents, make bline False.
    bline = True

    # Add whichever sets of model paths you'd like to evaluate to pathways:
    # (Evaluate either bline or meander in separate runs)

    pathways = icm_b

    for model_path in pathways:

        agent = MainAgent(chkp=model_path)
        print(
            f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

        file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime(
            "%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
        # print(f'Saving evaluation results to {file_name}')

        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

        print(f'using CybORG v{cyborg_version}, {scenario}\n')
        for num_steps in [30, 50, 100]:
            if bline:
                agent_name = B_lineAgent
            else:
                agent_name = RedMeanderAgent
            for red_agent in [agent_name]:

                cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
                wrapped_cyborg = wrap(cyborg)

                observation = wrapped_cyborg.reset()
                # observation = cyborg.reset().observation

                action_space = wrapped_cyborg.get_action_space(agent_name)
                # action_space = cyborg.get_action_space(agent_name)
                total_reward = []
                actions = []
                for i in range(MAX_EPS):
                    r = []
                    a = []
                    # cyborg.env.env.tracker.render()
                    for j in range(num_steps):
                        action = agent.get_action(observation, action_space)
                        observation, reward_adjusted, reward_real, done, info = wrapped_cyborg.step(action)
                        # result = cyborg.step(agent_name, action)
                        r.append(reward_real)
                        # r.append(result.reward)
                        a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                    agent.end_episode()
                    total_reward.append(sum(r))
                    actions.append(a)
                    # observation = cyborg.reset().observation
                    observation = wrapped_cyborg.reset()
                print(
                    f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
                with open(file_name, 'a+') as data:
                    data.write(
                        f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                    for act, sum_rew in zip(actions, total_reward):
                        data.write(f'actions: {act}, total reward: {sum_rew}\n')
