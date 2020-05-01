import time

import multiagent.scenarios as scenarios
from Agent import Agent
from Cste import *
from Network import *
from multiagent.environment import MultiAgentEnv


def test(beta_pl, beta_op, n_episodes=10000, display=False):
    env = make_env()
    net = Network(14, 16)
    target_net = Network(14, 16)
    net.load_state_dict(torch.load('data/' + str(beta_pl) + '_' + str(beta_op) + '/net_45000'))
    target_net.load_state_dict(torch.load('data/' + str(beta_pl) + '_' + str(beta_op) + '/target_45000'))
    players = Agent(0, beta_pl, beta_op, net, target_net, None, epsilon=0)
    rewards = 0
    for i in range(n_episodes):
        ob = env.reset()
        ob = torch.unsqueeze(torch.tensor(ob[0], device=device), 0)

        cumulative_reward = 0
        for step in range(max_steps_episode):
            if display:
                env.render('1')
                time.sleep(0.1)

            actions = players.choose_action(ob)
            ob, reward, done_n, info = env.step(actions)
            ob = torch.unsqueeze(torch.tensor(ob[0], device=device), 0)

            reward = -reward[1]
            cumulative_reward += reward

            if all(done_n): break

        rewards += cumulative_reward

    print('average reward', rewards / n_episodes)


def make_env(env_name='love_chase'):
    scenario = scenarios.load(env_name + '.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_input = True
    return env

