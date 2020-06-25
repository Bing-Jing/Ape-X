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
from batchrecoder_AQL import BatchRecorder
import copy
import utils
class train_DQN():
    def __init__(self, env_id, max_step = 1e6, prior_alpha = 0.6, prior_beta_start = 0.4,
                    publish_param_interval=5, device = "cuda:0", n_steps=1,
                    batch_size = 32, gamma = 0.99, target_update_interval=20, save_interval = 200,
                    propose_sample=1, uniform_sample = 50, action_var = 0.25, ent_lam = 0.8, n_workers=10):
        self.prior_beta_start = prior_beta_start
        self.max_step = int(max_step)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.publish_param_interval = publish_param_interval
        self.save_interval = save_interval
        self.ent_lam = ent_lam
        self.lr = 1e-3
        self.n_workers = n_workers
        self.n_steps = n_steps

        self.device = device
        self.env = gym.make(env_id)
        self.model = AQL(env = self.env, propose_sample=propose_sample, uniform_sample = uniform_sample,
                             action_var = action_var, device=self.device).to(self.device)
        self.target_model = AQL(env = self.env, propose_sample=propose_sample, uniform_sample = uniform_sample,
                             action_var = action_var,device=self.device).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())


        self.replay_buffer = CustomPrioritizedReplayBuffer_AQL(1e7,alpha=prior_alpha)
        
        self.optimizer_q = optim.Adam(self.model.q.parameters(), self.lr)
        self.optimizer_proposal = optim.Adam(self.model.proposal.parameters(), self.lr)
        
        self.writer = SummaryWriter(comment="-{}-learner".format(self.env.unwrapped.spec.id))
        self.recoder = BatchRecorder(env_id, env_seed=0, n_workers=n_workers, buffer=self.replay_buffer, 
                                        max_episode_length=50000, writer=self.writer, n_steps=n_steps, gamma=gamma,
                                        propose_sample=propose_sample, uniform_sample = uniform_sample, 
                                        action_var = action_var, device = self.device)

        # decay function
        self.scheduler_q = optim.lr_scheduler.StepLR(self.optimizer_q,step_size=100,gamma=0.99)
        self.scheduler_proposal = optim.lr_scheduler.StepLR(self.optimizer_proposal,step_size=100,gamma=0.99)

        self.beta_by_frame = lambda frame_idx: min(1.0, self.prior_beta_start + frame_idx * (1.0 - self.prior_beta_start) / self.max_step*self.n_workers)
        
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
        self.update_target(self.model.proposal, self.target_model.proposal)
        # self.scheduler_proposal.step()

        loss_q, prios = utils.compute_loss_AQL(self.model, self.target_model, batch, n_steps=self.n_steps,gamma=self.gamma)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.q.parameters(), 40)
        
        self.replay_buffer.update_priorities(indices, prios)
        self.optimizer_q.step()
        # self.scheduler_q.step()

        self.model.reset_noise()
        self.target_model.reset_noise()

        
        return loss_q, loss_p
    def train(self):
        
        learn_idx = 0
        for frame_idx in range(self.max_step):
            self.model.q.train()
            self.target_model.q.train()
            self.recoder.set_worker_weights(copy.deepcopy(self.model))
            
            total_ep = self.recoder.record_batch()
            for _ in range(total_ep//self.batch_size):
                
                if len(self.replay_buffer) > self.batch_size:
                    beta = self.beta_by_frame(frame_idx)
                    loss_q, loss_p = self.compute_td_loss(self.batch_size, beta)
                    self.writer.add_scalar("learner/loss_q", loss_q, learn_idx)
                    self.writer.add_scalar("learner/loss_proposal", loss_p, learn_idx)
                    
                learn_idx += 1
            if frame_idx % self.target_update_interval == 0:
                print("update target...")
                self.update_target(self.model, self.target_model)

            if frame_idx % self.save_interval == 0 or frame_idx == self.max_step-1:
                print("save model...")
                self.save_model(frame_idx)

        self.recoder.cleanup()
    def save_model(self, idx):
        torch.save(self.model.state_dict(), "./model{}.pth".format(idx))
    def load_model(self,idx):
         with open("model{}.pth".format(idx), "rb") as f:
                print("loading weights_{}".format(idx))
                self.model.load_state_dict(torch.load(f,map_location="cpu"))

training = True
if __name__ == "__main__":
    env_id = "CartPole-v0"#"BipedalWalker-v3"#"BipedalWalker-v3"#"LunarLanderContinuous-v2"#"LunarLander-v2"#"MountainCarContinuous-v0"#
   
    if training:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        test = train_DQN(env_id=env_id,device=device)
        test.train()
    else:
        device = torch.device("cpu")
        test = train_DQN(env_id=env_id,device=device)
        test.load_model(1600)
        test.model.q.eval()
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
        test.recoder.cleanup()
        test.env.close()
    