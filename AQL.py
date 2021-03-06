import math, random

import gym
# import pybulletgym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from model import AQL
from memory import CustomPrioritizedReplayBuffer_AQL
from tensorboardX import SummaryWriter
import utils

class train_DQN():
    def __init__(self, env_id, max_step = 1e6, prior_alpha = 0.6, prior_beta_start = 0.4, 
                    epsilon_start = 1, epsilon_final = 0.01, epsilon_decay = 1e4,
                    batch_size = 32, gamma = 0.99, target_update_interval=1000, save_interval = 1e4,
                    propose_sample=100, uniform_sample = 100, action_var = 0.25, ent_lam = 0.8):
        self.prior_beta_start = prior_beta_start
        self.max_step = int(max_step)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.save_interval = save_interval
        self.ent_lam = ent_lam
        self.lr = 1e-4

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(env_id)
        self.model = AQL(env = self.env, propose_sample=propose_sample, uniform_sample = uniform_sample,
                             action_var = action_var, device=self.device).to(self.device)
        self.target_model = AQL(env = self.env, propose_sample=propose_sample, uniform_sample = uniform_sample,
                             action_var = action_var,device=self.device).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())


        self.replay_buffer = CustomPrioritizedReplayBuffer_AQL(100000,alpha=prior_alpha)
        self.optimizer_q = optim.Adam(self.model.q.parameters(), self.lr)
        self.optimizer_proposal = optim.Adam(self.model.proposal.parameters(), self.lr)
        
        self.writer = SummaryWriter(comment="-{}-learner".format(self.env.unwrapped.spec.id))


        # decay function
        self.scheduler_q = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_q,T_max=self.max_step,eta_min=self.lr/1000)
        self.scheduler_proposal = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_proposal,T_max=self.max_step,eta_min=self.lr/1000)

        self.beta_by_frame = lambda frame_idx: min(1.0, self.prior_beta_start + frame_idx * (1.0 - self.prior_beta_start) / self.max_step)
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        
    def update_target(self,current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    def compute_td_loss(self,batch_size, beta):
        state, action, reward, next_state, done, a_mu, weights, indices  = self.replay_buffer.sample(batch_size, beta) 
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.LongTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)
        weights    = torch.FloatTensor(weights).to(self.device)
        a_mu       = torch.FloatTensor(a_mu).to(self.device)
        # reward = ((reward - reward.mean()) / (reward.std() + 1e-5))*2 -1
        batch = (state, action, reward, next_state, done, a_mu, weights)

        
        q_values      = self.model(state, a_mu)
        # next_q_values = self.target_model(next_state, a_mu)
        # q_value = q_values[torch.arange(batch_size), action].to(self.device)
        
        # next_q_value     = next_q_values.max(1)[0].to(self.device)
        # expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        # td_error = torch.abs(expected_q_value.detach() - q_value)
        
        # loss_q  = (td_error).pow(2) * weights
        # prios = loss_q+1e-5#0.9 * torch.max(td_error)+0.1*td_error+1e-5
        # loss_q  = loss_q.mean()
        loss_q, prios = utils.compute_loss_AQL(self.model, self.target_model, batch, n_steps=1,gamma=self.gamma)
        

        embed_state = self.model.q.embedding_feature(state)
        dist = self.model.proposal.evaluate(embed_state)
        max_q_action = a_mu[torch.arange(batch_size), q_values.max(1)[1]].reshape(batch_size,-1).to(self.device)
        log_prob = dist.log_prob(max_q_action)

        entropy = dist.entropy()
        loss_p = -log_prob - self.ent_lam * entropy
        loss_p = torch.mean(loss_p)
        self.optimizer_proposal.zero_grad()
        loss_p.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.proposal.parameters(), 40)
        
        self.optimizer_proposal.step()
        self.scheduler_proposal.step()

        self.optimizer_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.q.parameters(), 40)
        
        self.replay_buffer.update_priorities(indices, prios)
        self.optimizer_q.step()
        self.scheduler_q.step()
        # self.update_target(self.model.proposal, self.target_model.proposal)
        
        return loss_q, loss_p
    def train(self):
        episode_reward = 0
        episode_idx = 0
        episode_length = 0
        state = self.env.reset()
        for frame_idx in range(self.max_step):
            # epsilon = self.epsilon_by_frame(frame_idx)
            epsilon = 0.5 if random.random() > 0.1 else 0.05 # 10% actor epsilon = 0.5

            action, a_mu, _ = self.model.act((state), epsilon)
            a_mu = a_mu[0]
            next_state, reward, done, _ = self.env.step(a_mu[action])
            self.replay_buffer.add(state, action, reward, next_state, done, a_mu)
            
            state = next_state
            episode_reward += reward
            
            episode_length += 1
            if done:
                state = self.env.reset()
                self.writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
                self.writer.add_scalar("actor/episode_length", episode_length, episode_idx)
                # print("episode: ",episode_idx, " reward: ", episode_reward)
                episode_reward = 0
                episode_length = 0
                episode_idx += 1
                
            if len(self.replay_buffer) > self.batch_size:
                beta = self.beta_by_frame(frame_idx)
                loss_q, loss_p = self.compute_td_loss(self.batch_size, beta)
                self.writer.add_scalar("learner/loss_q", loss_q, frame_idx)
                self.writer.add_scalar("learner/loss_proposal", loss_p, frame_idx)
                
            if frame_idx % self.target_update_interval == 0:
                print("update target...")
                self.update_target(self.model, self.target_model)

            if frame_idx % self.save_interval == 0 or frame_idx == self.max_step-1:
                print("save model...")
                self.save_model(frame_idx)

        self.env.close()
    def save_model(self, idx):
        torch.save(self.model.state_dict(), "./model{}.pth".format(idx))
    def load_model(self,idx):
         with open("model{}.pth".format(idx), "rb") as f:
                print("loading weights_{}".format(idx))
                self.model.load_state_dict(torch.load(f,map_location="cpu"))

training = False
if __name__ == "__main__":
    env_id = "Pendulum-v0"

    test = train_DQN(env_id=env_id)
    if training:
        test.train()
    else:
        # test.device = "cpu"
        # test.model.to("cpu")
        test.load_model(200)
        for i in range(10):
            # test.env.render()
            s = test.env.reset()
            er = 0
            d = False
            while True:
                test.env.render(mode='rgb_array')
                a, a_mu,_ = test.model.act(s, epsilon=0)
                s, r, d, _ = test.env.step(a_mu[0][a])
                er+=r
                if d:
                    print(er)
                    break
        test.env.close()
    