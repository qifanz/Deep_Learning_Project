import pickle
import time

import multiagent.scenarios as scenarios
from Agent import Agent
from Cste import *
from Network import *
from ReplayBuffer import ReplayBuffer
from multiagent.environment import MultiAgentEnv


def train(beta_pl, beta_op):
    print('Start training...')
    replay_buffer = ReplayBuffer(replay_buffer_size)
    env = make_env()
    net = Network(14, 16)
    target_net = Network(14, 16)
    if device == 'cuda':
        net = net.cuda()
        target_net = target_net.cuda()

    players = Agent(0, beta_pl, beta_op , net, target_net,replay_buffer)
    rewards_list = []
    bellman_errors_list = []

    sum_cumulative_reward = 0
    sum_cumulative_error = 0

    for i in range(num_episodes):
        ob = env.reset()
        ob = torch.unsqueeze(torch.tensor(ob[0], device=device), 0)

        cumulative_reward = 0
        cumulative_error = 0
        for step in range(max_steps_episode):
            if display_env:
                env.render('1')
                time.sleep(0.1)

            previous_ob = ob.clone().detach()
            action_pl, action_op, action_tensor = players.choose_action(ob)
            ob, rewards, done_n, info = env.step([action_pl, action_op])
            ob = torch.unsqueeze(torch.tensor(ob[0], device=device), 0)
            reward = -rewards[1]
            # print(reward)
            cumulative_reward += reward
            #error = players.update(previous_ob, reward, actions, ob)
            if all(done_n): break
            replay_buffer.push(previous_ob, reward, action_tensor, ob)
            error = players.optimize()
            cumulative_error += error

        rewards_list.append(cumulative_reward)
        bellman_errors_list.append(cumulative_error)
        sum_cumulative_reward += cumulative_reward
        sum_cumulative_error += cumulative_error

        if i % print_frequency == 0 and i != 0:
            print('episode ', i, sum_cumulative_reward / print_frequency, sum_cumulative_error / print_frequency)
            sum_cumulative_error = 0
            sum_cumulative_reward = 0
        if i % save_frequency == 0 and i != 0:
            net_weights_path = data_dir + 'net_' + str(i)
            target_weights_path = data_dir + 'target_' + str(i)

            torch.save(net.state_dict(), net_weights_path)
            torch.save(target_net.state_dict(), target_weights_path)
            f = open('data/rewards.pkl', 'wb')
            pickle.dump(rewards_list, f)
            f.close()
            f = open('data/errors.pkl', 'wb')
            pickle.dump(bellman_errors_list, f)
            f.close()



def make_env(env_name='love_chase'):
    scenario = scenarios.load(env_name + '.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_input = True
    return env


