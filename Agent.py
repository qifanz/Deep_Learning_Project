import math

import numpy as np
import torch
import torch.nn as nn
from Cste import *
from ReplayBuffer import *
class Agent:
    def __init__(self, index, beta_pl, beta_op, Q_net, Q_target, replay_buffer, num_actions=4, epsilon=epsilon_greedy):
        self.gamma = 0.9
        self.index = index
        self.beta_pl = beta_pl
        self.beta_op = beta_op
        self.Q_net = Q_net
        self.Q_target = Q_target
        self.num_actions = num_actions
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=0.0005)
        self.loss_function = nn.MSELoss()
        self.learn_step = 0
        self.epsilon=epsilon
        self.replay_buffer = replay_buffer

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            player_action = 1 + np.random.randint(self.num_actions)
            opponent_action =  1 + np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                action_values = self.Q_net.forward(observation).cpu().numpy()[0]
                player_action =  1 + self._get_player_action(observation, action_values)
                opponent_action = 1 + self._get_opponent_action(observation,action_values)

        action_tensor = torch.unsqueeze(torch.tensor([self._get_combined_index(player_action - 1, opponent_action - 1)],
                                     dtype=torch.int64, device=device),0)

        return player_action, opponent_action, action_tensor

    def update(self, state, reward, actions, new_state):
        self.learn_step += 1
        if self.learn_step % target_update == 0:
            self.Q_target.load_state_dict(self.Q_net.state_dict())
        Q_estimate = reward + self.gamma * self._compute_estimated_value(new_state)
        action_tensor = torch.tensor([[self._get_combined_index(actions[0]-1, actions[1]-1)]], dtype=torch.int64, device=device)
        Q_current = self.Q_net(state).gather(1, action_tensor)
        loss = self.loss_function(Q_current,torch.tensor([[Q_estimate]], device=device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().data.numpy()

    def optimize(self):
        self.learn_step += 1
        if self.learn_step % target_update == 0:
            self.Q_target.load_state_dict(self.Q_net.state_dict())

        if len(self.replay_buffer) < batch_size: return 0
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # compute estimates tensor
        rewards = batch.reward
        new_states = batch.next_state
        Q_estimates = []
        for i in range(len(rewards)):
            Q_estimate = rewards[i] + self.gamma * self._compute_estimated_value(new_states[i])
            Q_estimates.append([Q_estimate])
        Q_estimates = torch.tensor(Q_estimates, device=device)

        #compute Q value tensor
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.actions)
        Q_values = self.Q_net(state_batch).gather(1, action_batch)

        #optimize
        loss = self.loss_function(Q_values,Q_estimates)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.sum(loss.cpu().data.numpy())






    def _compute_estimated_value(self, observation):
        with torch.no_grad():
            action_values = self.Q_target(observation).detach().cpu().numpy()[0]
            value = 0
            for action in range(self.num_actions):
                value += self._get_reference_pl(observation, action) * math.exp(
                    self.beta_pl * self._marginalize_pl(observation, action_values, action))
            value = math.log(value) / self.beta_pl
            return value

    def _get_player_action(self, observation, action_values):
        policy = []
        for action in range(self.num_actions):
            Q_pl = self._marginalize_pl(observation, action_values, action)
            prob = self._get_reference_pl(observation, action) * math.exp(self.beta_pl * Q_pl)
            policy.append(prob)
        policy = self._normalize(policy)
        action_chosen = np.random.choice(np.arange(self.num_actions), p=policy)
        return action_chosen

    def _marginalize_pl(self, observation, action_values, action_pl):
        Q_pl = 0
        for action_op in range(self.num_actions):
            index = self._get_combined_index(action_pl, action_op)
            Q_pl += self._get_reference_op(observation, action_op) * math.exp(self.beta_op * action_values[index])
        Q_pl = math.log(Q_pl) / self.beta_op
        return Q_pl

    def _marginalize_op(self, observation, action_values, action_op):
        Q_op = 0
        for action_pl in range(self.num_actions):
            index = self._get_combined_index(action_pl, action_op)
            Q_op += self._get_reference_pl(observation, action_pl) * math.exp(self.beta_pl * action_values[index])
        Q_op = math.log(Q_op) / self.beta_pl
        return Q_op

    def _get_opponent_action(self, observation, action_values):
        policy = []
        for action in range(self.num_actions):
            Q_op = self._marginalize_op(observation, action_values, action)
            prob = self._get_reference_op(observation, action) * math.exp(self.beta_op * Q_op)
            policy.append(prob)
        policy = self._normalize(policy)
        action_chosen = np.random.choice(np.arange(self.num_actions), p=policy)
        return action_chosen

    def _get_combined_index(self, index_pl, index_op):
        return index_pl * self.num_actions + index_op

    def _get_reference_pl(self, observation, action):
        return 1 / self.num_actions

    def _get_reference_op(self, observation, action):
        return 1 / self.num_actions

    def _normalize(self, array):
        return np.divide(array, np.sum(array))
