import pickle
import time

import multiagent.scenarios as scenarios
from Agent import Agent
from Cste import *
from Network import *
from multiagent.environment import MultiAgentEnv


def train():
    print('Start training...')
    env = make_env()
    net = Network(14, 16)
    target_net = Network(14, 16)
    if device == 'gpu':
        net = net.cuda()
        target_net = target_net.cuda()

    players = Agent(0, 20, -20, net, target_net)
    rewards = []
    bellman_errors = []

    sum_cumulative_reward = 0
    sum_cumulative_error = 0

    for i in range(num_episodes):
        ob = env.reset()[0]
        cumulative_reward = 0
        cumulative_error = 0
        for step in range(max_steps_episode):
            if display_env:
                env.render('1')
                time.sleep(0.1)

            previous_ob = ob.copy()
            actions = players.choose_action(ob)
            ob, rewards, done_n, info = env.step(actions)
            ob = ob[0]
            reward = -rewards[1]
            # print(reward)
            cumulative_reward += reward
            error = players.update(previous_ob, reward, actions, ob)
            cumulative_error += error
            if all(done_n): break

        rewards.append(cumulative_reward)
        bellman_errors.append(cumulative_error)
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
            pickle.dump(rewards, f)
            f.close()
            f = open('data/errors.pkl', 'wb')
            pickle.dump(bellman_errors, f)
            f.close()


def make_env(env_name='love_chase'):
    scenario = scenarios.load(env_name + '.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_input = True
    return env


train()
