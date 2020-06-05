"""
Module for DQN Model in Ape-X.
"""
import random
import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Categorical, MultivariateNormal, uniform
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env):
        super(DuelingDQN, self).__init__()

        self.input_shape = env.observation_space.shape
        if len(self.input_shape) == 3:
            self.cnn = True
        else:
            self.cnn = False

        self.num_actions = env.action_space.n

        self.flatten = Flatten()
        if self.cnn:
            self.features = nn.Sequential(
                init(nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU()
            )
        else:
            self.features = nn.Sequential(
                init(nn.Linear(self.input_shape[0],128)),
                nn.ReLU(),

            )
        # self.l1 = nn.Linear(self._feature_size(), 512)
        # self.adv = nn.Linear(512, self.num_actions)
        self.advantage = nn.Sequential(
            init(nn.Linear(self._feature_size(), 128)),
            nn.ReLU(),
            init(nn.Linear(128, self.num_actions))
        )

        self.value = nn.Sequential(
            init(nn.Linear(self._feature_size(), 128)),
            nn.ReLU(),
            init(nn.Linear(128, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        # x = self.l1(x)
        # advantage = self.adv(x)
        # print(advantage)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)
        # return torch.ones((1,self.num_actions))

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def act(self, state, epsilon):
        """
        Return action, max_q_value for given state
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = self.forward(state)

            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return action, q_values.cpu().numpy()[0]


class Flatten(nn.Module):
    """
    Simple module for flattening parameters
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


def init_(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init(module):
    return init_(module,
                 nn.init.orthogonal_,
                 lambda x: nn.init.constant_(x, 0),
                 nn.init.calculate_gain('relu'))





################################

class AQL(nn.Module):
    def __init__(self, env, propose_sample=100, uniform_sample = 400, action_var = 0.25):
        super(AQL, self).__init__()
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.env_iscontinuous = isinstance(self.env.action_space, gym.spaces.Box)
        self.total_sample = propose_sample+uniform_sample
        if self.env_iscontinuous:
            self.num_actions = env.action_space.shape[0]
        else:
            self.num_actions = env.action_space.n

        self.q = Q_Network(input_shape = self.input_shape, num_actions= self.num_actions, total_sample = self.total_sample,
                            env_iscontinuous = self.env_iscontinuous)
        
        self.proposal = Proposal_Network(self.env, propose_sample=propose_sample, uniform_sample = uniform_sample, action_var = action_var)

    def forward(self, state, epsilon):
        x = self.q.embedding_feature(state)
    
        a_mu = self.proposal.forward(x)

        action, q_values = self.q.act(state, a_mu, epsilon)
        return action, a_mu.cpu().numpy(), q_values

    def act(self, state, epsilon):
        with torch.no_grad():
            x = self.q.embedding_feature(state)
            a_mu = self.proposal.forward(x)

            action, q_values = self.q.act(state, a_mu, epsilon)
            return action, a_mu.cpu().numpy(), q_values

class Q_Network(nn.Module):
    def __init__(self, input_shape, num_actions, total_sample, env_iscontinuous):
        super(Q_Network, self).__init__()

        self.input_shape = input_shape
        self.total_sample = total_sample
        if len(self.input_shape) == 3:
            self.cnn = True
        else:
            self.cnn = False

        self.num_actions = num_actions
        self.env_iscontinuous = env_iscontinuous

        self.a_out_unit = 300
        self.feature_out_unit = 300
        self.concat_unit = self.a_out_unit + self.feature_out_unit

        if self.cnn:
            self.features = nn.Sequential(
                init(nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU()
            )
        else:
            self.features = nn.Sequential(
                init(nn.Linear(self.input_shape[0],128)),
                nn.ReLU(),
            )

        
        self.q_feature = nn.Sequential(
                init(nn.Linear(self.input_shape[0],400)),
                nn.ReLU(),
                init(nn.Linear(400,self.feature_out_unit))
            )
        
        self.action_out = nn.Linear(self.num_actions*self.total_sample, self.a_out_unit) #????????????????

        self.advantage = nn.Sequential(
            init(nn.Linear(self.concat_unit, 128)),
            nn.ReLU(),
            init(nn.Linear(128, self.total_sample))
        )

        self.value = nn.Sequential(
            init(nn.Linear(self.concat_unit, 128)),
            nn.ReLU(),
            init(nn.Linear(128, 1))
        )

    def forward(self, x):
            x = x.reshape(-1,self.concat_unit)
            advantage = self.advantage(x)
            value = self.value(x)
            return value + advantage - advantage.mean(1, keepdim=True)

    def embedding_feature(self, x):
            x = self.features(x)
            x = x.reshape(-1,128)
            return x

    
    def act(self, state, a_mu, epsilon):
            print(a_mu.shape)
            a_mu = a_mu.reshape(-1,self.total_sample, self.num_actions)
            a_feature = F.softmax(a_mu).reshape(a_mu.shape[0],-1) # ???????????????????????

            q_f = self.q_feature(state).reshape(-1, self.feature_out_unit)
            a_out = self.action_out(a_feature)
            # print(q_f.shape, a_out.shape)
            
            x = torch.cat([a_out,q_f],dim=1)
            x = F.relu(x)
            q_values = self.forward(x)
            # print(q_values.shape)

            if random.random() > epsilon:
                idx = torch.argmax(q_values,dim=1)
                action = idx[0]
            else:
                action = random.choice(list(range(a_mu[0].shape[0])))
            return action, q_values

class Proposal_Network(nn.Module):
    def __init__(self, env, propose_sample=100, uniform_sample = 100, action_var = 0.25):
        super(Proposal_Network, self).__init__()
        self.env = env
        self.input_shape = env.observation_space.shape
        self.env_iscontinuous = isinstance(self.env.action_space, gym.spaces.Box)
        self.uniform_sample = uniform_sample
        self.propose_sample = propose_sample
        if self.env_iscontinuous:
            self.num_actions = env.action_space.shape[0]
        else:
            self.num_actions = env.action_space.n
        
        self.dist_feature = nn.Sequential(
                init(nn.Linear(128,300)),
                nn.ReLU(),
                init(nn.Linear(300,200)),
                nn.ReLU(),
                init(nn.Linear(200,self.num_actions))
            )
        self.action_out_feature = nn.Linear(self.num_actions,300)
        self.action_var = torch.full((self.num_actions,), action_var)
        if self.env_iscontinuous:
            self.uniform = uniform.Uniform(torch.Tensor(self.env.action_space.low),torch.Tensor(self.env.action_space.high))

    def forward(self,embed_state):
            mu = self.dist_feature(embed_state)
            if self.env_iscontinuous: # continuous
                cov_mat = torch.diag(self.action_var)
                dist = MultivariateNormal(mu, cov_mat)
                a_uniform = self.uniform.sample([mu.shape[0],self.uniform_sample])
                a_dist = dist.sample([self.propose_sample]).reshape((-1,self.propose_sample,self.num_actions))
                a_mu = torch.cat([a_uniform,a_dist],dim=1)
            else:  # discrete
                dist = Categorical(logits=mu)
                a_mu = dist.sample([self.num_actions])

            return a_mu

    def evaluate(self, embed_state):
            mu = self.dist_feature(embed_state)
            if self.env_iscontinuous: # continuous
                cov_mat = torch.diag(self.action_var)
                dist = MultivariateNormal(mu, cov_mat)
                a_uniform = self.uniform.sample([self.uniform_sample])
                a_dist = dist.sample([self.propose_sample]).reshape((-1,self.num_actions))
                a_mu = torch.cat([a_uniform,a_dist])
            else:  # discrete
                dist = Categorical(logits=mu)
                a_mu = dist.sample([self.num_actions])

            return a_mu, dist
        
        