import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from model import AQL
from memory import PrioritizedReplayBuffer
from tensorboardX import SummaryWriter

class train_DQN():
    def __init__(self, env_id, max_step = 1e6, prior_alpha = 0.6, prior_beta_start = 0.4, 
                    epsilon_start = 1.0, epsilon_final = 0.01, epsilon_decay = 500,
                    batch_size = 32, gamma = 0.99, target_update_interval=1000, save_interval = 10000,
                    propose_sample=100, uniform_sample = 400, action_var = 0.25, ent_lam = 0.8):
        self.prior_beta_start = prior_beta_start
        self.max_step = int(max_step)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.save_interval = save_interval
        self.ent_lam = ent_lam

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(env_id)
        self.model = AQL(env = self.env, propose_sample=propose_sample, uniform_sample = uniform_sample,
                             action_var = action_var, device=self.device).to(self.device)
        self.target_model = AQL(env = self.env, propose_sample=propose_sample, uniform_sample = uniform_sample,
                             action_var = action_var,device=self.device).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())


        self.replay_buffer = PrioritizedReplayBuffer(100000,alpha=prior_alpha)
        self.optimizer_q = optim.Adam(self.model.q.parameters())
        self.optimizer_proposal = optim.Adam(self.model.proposal.parameters())
        
        self.writer = SummaryWriter(comment="-{}-learner".format(self.env.unwrapped.spec.id))


        # decay function
        self.scheduler_q = optim.lr_scheduler.StepLR(self.optimizer_q,step_size=1000,gamma=0.99)
        self.scheduler_proposal = optim.lr_scheduler.StepLR(self.optimizer_q,step_size=1000,gamma=0.99)

        self.beta_by_frame = lambda frame_idx: min(1.0, self.prior_beta_start + frame_idx * (1.0 - self.prior_beta_start) / 1000)
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
        
    def update_target(self,current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    def compute_td_loss(self,batch_size, beta):
        state, action, reward, next_state, done, weights, indices  = self.replay_buffer.sample(batch_size, beta) 
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.LongTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).to(self.device)
        done       = torch.FloatTensor(done).to(self.device)
        weights    = torch.FloatTensor(weights).to(self.device)
        
        _, a_mu, q_values      = self.model(state, 0)
        _, _, next_q_values = self.target_model(next_state, 0)

        q_value = q_values[torch.arange(batch_size), action]

        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        td_error = torch.abs(expected_q_value.detach() - q_value)
        loss_q  = (td_error).pow(2) * weights
        prios = loss_q+1e-5#0.9 * torch.max(td_error)+(1-0.9)*td_error
        loss_q  = loss_q.mean()
        
        embed_state = self.model.q.embedding_feature(state)
        _, dist = self.model.proposal.evaluate(embed_state)
        max_q_action = []
        for i in range(batch_size):
            max_q_action.append(a_mu[0][q_values.max(1)[1][i]])
        max_q_action = torch.FloatTensor(max_q_action).to(self.device)
        log_prob = dist.log_prob(max_q_action)

        entropy = dist.entropy()
        loss_p = -log_prob - self.ent_lam * entropy
        loss_p = torch.mean(loss_p)
        self.optimizer_proposal.zero_grad()
        loss_p.backward()
        self.scheduler_proposal.step()
        self.optimizer_proposal.step()

        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.scheduler_q.step()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer_q.step()
        
        return loss_q, loss_p
    def train(self):
        losses = []
        all_rewards = []
        episode_reward = 0
        episode_idx = 0
        episode_length = 0
        state = self.env.reset()
        for frame_idx in range(self.max_step):
            epsilon = self.epsilon_by_frame(frame_idx)
            action, a_mu, _ = self.model.act(torch.FloatTensor((state)).to(self.device), epsilon)
            next_state, reward, done, _ = self.env.step(a_mu[0][action])
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            episode_length += 1
            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                self.writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
                self.writer.add_scalar("actor/episode_length", episode_length, episode_idx)
                # print("episode: ",episode_idx, " reward: ", episode_reward)
                episode_reward = 0
                episode_length = 0
                episode_idx += 1
                
            if len(self.replay_buffer) > self.batch_size:
                beta = self.beta_by_frame(frame_idx)
                loss_q, loss_p = self.compute_td_loss(self.batch_size, beta)
                losses.append(loss_q.item())
                self.writer.add_scalar("learner/loss_q", loss_q, frame_idx)
                self.writer.add_scalar("learner/loss_proposal", loss_p, frame_idx)
                
            if frame_idx % self.target_update_interval == 0:
                print("update target...")
                self.update_target(self.model, self.target_model)

            if frame_idx % self.save_interval == 0 or frame_idx == self.max_step-1:
                print("save model...")
                self.save_model(frame_idx)

            
    def save_model(self, idx):
        torch.save(self.model.state_dict(), "./model{}.pth".format(idx))
    def load_model(self,idx):
         with open("model{}.pth".format(idx), "rb") as f:
                print("loading weights_{}".format(idx))
                self.model.load_state_dict(torch.load(f,map_location="cpu"))

training = True
if __name__ == "__main__":
    env_id = "LunarLanderContinuous-v2"

    test = train_DQN(env_id=env_id)
    if training:
        test.train()
    else:
        test.device = "cpu"
        test.model.to("cpu")
        test.load_model(99999)
        for i in range(10):
            s = test.env.reset()
            s = torch.FloatTensor(s)
            er = 0
            d = False
            while True:
                test.env.render()
                a,_ = test.model.act(s, epsilon=0)
                s, r, d, _ = test.env.step(a)
                er+=r
                s = torch.FloatTensor(s)
                if d:
                    print(er)
                    break
        test.env.close()
    