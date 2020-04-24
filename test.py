import time

import multiagent.scenarios as scenarios
from Agent import Agent
from Cste import *
from Network import *
from multiagent.environment import MultiAgentEnv

def t():
    env = make_env()
    net = Network(14, 16)
    target_net = Network(14, 16)
    net.load_state_dict(torch.load('data/net_100000'))
    target_net.load_state_dict(torch.load('data/target_100000'))
    players = Agent(0, 20, -20, net, target_net)
    rewards = []
    for i in range(num_episodes):
        ob = env.reset()[0]
        cumulative_reward = 0
        for step in range(max_steps_episode):
            env.render('1')
            time.sleep(0.1)

            actions = players.choose_action(ob)
            ob, reward, done_n, info = env.step(actions)
            ob = ob[0]
            reward = -reward[1]
            cumulative_reward += reward

            if all(done_n): break

        rewards.append(cumulative_reward)

        if i % 1 == 0:
            print('episode ', i, cumulative_reward)


def make_env(env_name='love_chase'):
    scenario = scenarios.load(env_name + '.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_input = True
    return env


t()
