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
import math
from torch.autograd import Variable

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




class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, device, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.device     = device
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        weight_epsilon = self.weight_epsilon.to(self.device)
        bias_epsilon   = self.bias_epsilon.to(self.device)
            
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


################################

class AQL(nn.Module):
    def __init__(self, env, propose_sample=100, uniform_sample = 400, action_var = 0.25, device = "cuda"):
        super(AQL, self).__init__()
        self.device = device
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.env_iscontinuous = isinstance(self.env.action_space, gym.spaces.Box)
        
        if self.env_iscontinuous:
            self.num_actions = env.action_space.shape[0]
            self.uniform_sample = uniform_sample
        else:
            if uniform_sample > env.action_space.n:
                self.uniform_sample = env.action_space.n
            else:
                self.uniform_sample = uniform_sample
            self.num_actions = env.action_space.n
        self.total_sample = propose_sample+self.uniform_sample
        self.q = Q_Network(input_shape = self.input_shape, num_actions= self.num_actions, total_sample = self.total_sample,
                            env_iscontinuous = self.env_iscontinuous, device=device)
        
        self.proposal = Proposal_Network(self.env, propose_sample=propose_sample,
                            uniform_sample = self.uniform_sample, action_var = action_var, device=self.device)

    def forward(self, state, a_mu):

        _, q_values = self.q.act(state, a_mu, 0)
        return q_values

    def act(self, state, epsilon):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            x = self.q.embedding_feature(state)
            a_mu = self.proposal.forward(x)

            action, q_values = self.q.act(state, a_mu, epsilon)
            return action, a_mu.cpu().numpy(), q_values
    def reset_noise(self):
        self.q.reset_noise()

class Q_Network(nn.Module):
    def __init__(self, input_shape, num_actions, total_sample, env_iscontinuous,device):
        super(Q_Network, self).__init__()
        self.device = device

        self.input_shape = input_shape
        self.total_sample = total_sample
        if len(self.input_shape) == 3:
            self.cnn = True
        else:
            self.cnn = False

        self.num_actions = num_actions
        self.env_iscontinuous = env_iscontinuous

        self.a_out_unit = 64
        self.feature_out_unit = 64
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
                init(nn.Linear(self.input_shape[0],64)),
                nn.ReLU(),
                init(nn.Linear(64,self.feature_out_unit)),
                nn.ReLU()
            )
        if self.env_iscontinuous:
            # self.action_out = nn.Linear(self.num_actions, self.a_out_unit)
            self.action_out = nn.Sequential(
                    nn.Linear(self.num_actions, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.a_out_unit),
                    nn.ReLU(),
            )
            
        else:
            self.action_out = nn.Sequential(
                nn.Linear(1, self.a_out_unit),
                nn.ReLU(),
            )
            # self.advantage1 = NoisyLinear(self.concat_unit, 64, device=self.device)
            # self.advantage2 = NoisyLinear(64, self.total_sample, device=self.device)
        self.advantage1 = NoisyLinear(self.concat_unit, 64, device=self.device)
        self.advantage2 = NoisyLinear(64, 1, device=self.device)
        # self.value1 = NoisyLinear(self.feature_out_unit, 64, device=self.device)
        # self.value2 = NoisyLinear(64, 1, device=self.device)
        
        

    def forward(self, x,q_f):
            x = x.reshape(-1,self.concat_unit)
            # x = self.concat_out(x)
            advantage = F.relu(self.advantage1(x))
            advantage = self.advantage2(advantage)

            # value = F.relu(self.value1(q_f))
            # value = self.value2(value)
            return advantage #+ value - advantage.mean(0, keepdim=True)

    def embedding_feature(self, x):
            x = self.features(x)
            x = x.reshape(-1,128)
            return x

    def reset_noise(self):
        # self.value1.reset_noise()
        # self.value2.reset_noise()
        self.advantage1.reset_noise()
        self.advantage2.reset_noise()

    def act(self, state, a_mu, epsilon):
            # print(a_mu.shape)
            if self.env_iscontinuous:
                ### ver 1
                a_mu = a_mu.reshape(-1,self.total_sample, self.num_actions)#32,200,2
                a_feature = a_mu.reshape(-1,self.num_actions)#32*200,2
                a_out = self.action_out(a_feature).reshape(-1,self.total_sample, self.a_out_unit)#32,200,64
                q_f = self.q_feature(state).repeat(1,self.total_sample).reshape(-1,self.total_sample, self.feature_out_unit)#32,200,64
                # print(q_f.shape,a_out.shape)
                x = torch.cat([a_out,q_f],dim=2)#32,200,128
                x = F.relu(x)
                q_values = self.forward(x,q_f).reshape(a_mu.shape[0], self.total_sample)
                # print(q_values.shape)

                ### ver2
                # a_mu = a_mu.reshape(-1,self.total_sample, self.num_actions)
                # a_feature = a_mu.reshape(-1,self.total_sample*self.num_actions)
                # a_out = self.action_out(a_feature).reshape(-1, self.a_out_unit)
                # q_f = self.q_feature(state).reshape(-1, self.feature_out_unit)
                # # print(q_f.shape,a_out.shape)
                # x = torch.cat([a_out,q_f],dim=1)#32,128
                # q_values = self.forward(x,q_f).reshape(a_mu.shape[0], self.total_sample)
                # # print(q_values.shape)
                
            else:
                a_feature = a_mu.reshape(-1,1).float()
                a_out = self.action_out(a_feature).reshape(-1, self.total_sample, self.a_out_unit)
                # print(a_out.shape)
                # q_f = self.q_feature(state).reshape(-1,self.feature_out_unit)
                q_f = self.q_feature(state).repeat(1,self.total_sample).reshape(-1,self.total_sample, self.feature_out_unit)#32,200,64
                # print(q_f.shape)
                x = torch.cat([a_out,q_f],dim=2)
                x = F.relu(x)
                # print(x.shape) 32,200,128
                q_values = self.forward(x,q_f).reshape(a_mu.shape[0], self.total_sample)
            # print(q_values.shape)
            if random.random() > epsilon:
                idx = torch.argmax(q_values,dim=1)
                action = idx[0].cpu().numpy()
            else:
                action = random.choice(list(range(self.total_sample)))
            return action, q_values

class Proposal_Network(nn.Module):
    def __init__(self, env, propose_sample=100, uniform_sample = 100, action_var = 0.25, device="cuda"):
        super(Proposal_Network, self).__init__()
        self.device = device
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
                init(nn.Linear(128,128)),
                nn.ReLU(),
                init(nn.Linear(128,self.num_actions)),
                # nn.Softmax(dim=1)
            )
        self.action_var = torch.full((self.num_actions,), action_var)
        if self.env_iscontinuous:
            self.uniform = uniform.Uniform(torch.Tensor(self.env.action_space.low),torch.Tensor(self.env.action_space.high))

    def forward(self,embed_state):
            mu = self.dist_feature(embed_state).to(self.device)
            if self.env_iscontinuous: # continuous
                cov_mat = torch.diag(self.action_var).to(self.device)
                dist = MultivariateNormal(mu, cov_mat)
                a_uniform = self.uniform.sample([mu.shape[0],self.uniform_sample]).to(self.device)
                a_dist = dist.sample([self.propose_sample]).reshape((-1,self.propose_sample,self.num_actions))
                a_mu = torch.cat([a_uniform,a_dist],dim=1)
            else:  # discrete
                dist = Categorical(logits=mu)
                a_dist = dist.sample([self.propose_sample]).reshape(mu.shape[0],self.propose_sample)
                a_uniform = np.random.choice(torch.arange(self.num_actions),size=self.uniform_sample,replace=False)
                a_uniform = torch.LongTensor(a_uniform).reshape(mu.shape[0],self.uniform_sample).to(self.device)
                a_mu = torch.cat([a_uniform,a_dist],dim=1)

            return a_mu

    def evaluate(self, embed_state):
            mu = self.dist_feature(embed_state)
            if self.env_iscontinuous: # continuous
                cov_mat = torch.diag(self.action_var).to(self.device)
                dist = MultivariateNormal(mu, cov_mat)
                # a_uniform = self.uniform.sample([mu.shape[0],self.uniform_sample]).to(self.device)
                # a_dist = dist.sample([self.propose_sample]).reshape((-1,self.propose_sample,self.num_actions)).to(self.device)
                # a_mu = torch.cat([a_uniform,a_dist],dim=1).to(self.device)
            else:  # discrete
                dist = Categorical(logits=mu)
                # a_mu = dist.sample([self.num_actions]).to(self.device)

            return dist
        
        