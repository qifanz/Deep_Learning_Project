import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

AGGREGATION_FACTOR = 500

LossTuple = namedtuple('Loss',
                       ('beta_pl', 'beta_op', 'loss'))


def plot_rewards(loss_tuples):
    plt.figure()
    title = 'Training curve'
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of loss in 1 episode')
    plt.ylim((0, 3))
    for loss_tuple in loss_tuples:
        plt.plot((smooth(loss_tuple.loss)),
                 label='beta_pl=' + str(loss_tuple.beta_pl) + ' beta_op=' + str(loss_tuple.beta_op))
    plt.legend()
    plt.show()


def plot_rewardss(nash_q, kl_uniform, kl_prior):
    plt.figure()
    title = 'Training curve of different algorithms'
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.ylim((-0.5, 0.5))
    plt.plot(smooth(nash_q), color='green', label='MiniMax-Q')
    plt.plot(smooth(kl_uniform), color='red', label='Soft Nash-Q2 with uniform prior')
    plt.plot(smooth(kl_prior), color='blue', label='Soft Nash-Q2 with quasi-nash prior')
    plt.legend()
    plt.show()


def smooth(rewards, factor=AGGREGATION_FACTOR):
    rewards = np.array(rewards)
    episodes = len(rewards)
    smoothen_rewards = np.zeros(episodes - factor)

    for i in range(0, episodes - factor):
        smoothen_rewards[i:i + factor] = np.mean(rewards[i:i + factor])
    return smoothen_rewards


loss_tuples = []
for beta_pl in [20, 10]:
    for beta_op in [-20, -10, -5]:
        f = open('data/' + str(beta_pl) + '_' + str(beta_op) + '/errors.pkl', 'rb')
        loss = pickle.load(f)
        f.close()
        loss_tuple = LossTuple(beta_pl, beta_op, loss)
        loss_tuples.append(loss_tuple)
plot_rewards(loss_tuples)
