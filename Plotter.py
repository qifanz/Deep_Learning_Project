import matplotlib.pyplot as plt
import numpy as np
import pickle
AGGREGATION_FACTOR = 1000



def plot_rewards(cumulative_rewards, algorithm):
    plt.figure()
    title = 'Reward - ' + algorithm
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    #plt.ylim((-0.5,0.5))
    plt.plot(smooth(cumulative_rewards))
    plt.show()

def plot_rewardss(nash_q,kl_uniform,kl_prior):
    plt.figure()
    title = 'Training curve of different algorithms'
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.ylim((-0.5,0.5))
    plt.plot(smooth(nash_q), color='green',label='MiniMax-Q')
    plt.plot(smooth(kl_uniform), color='red',label='Soft Nash-Q2 with uniform prior')
    plt.plot(smooth(kl_prior), color='blue',label='Soft Nash-Q2 with quasi-nash prior')
    plt.legend()
    plt.show()


def smooth(rewards):
    rewards = np.array(rewards)
    episodes = len(rewards)
    smoothen_rewards = np.zeros(episodes-AGGREGATION_FACTOR)

    for i in range(0, episodes - AGGREGATION_FACTOR):
        smoothen_rewards[i:i + AGGREGATION_FACTOR] = np.mean(rewards[i:i + AGGREGATION_FACTOR])
    return smoothen_rewards

f = open('rewards.pkl','rb')
rewards = pickle.load(f)
f.close()
plot_rewards(rewards, ' ')