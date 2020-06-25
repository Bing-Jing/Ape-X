import random
import gym
import numpy as np
import _pickle as pickle
import multiprocessing as mp
import torch
from model import AQL
from memory import BatchStorage
from tensorboardX import SummaryWriter
import itertools


class Worker(mp.Process):
    def __init__(self, worker_id, env_id, seed, epsilon, task_queue, buffer, max_episode_length,
                    propose_sample=100, uniform_sample = 400, action_var = 0.25, device = "cuda"):
        mp.Process.__init__(self)
        self.worker_id = worker_id
        self.env = gym.make(env_id)
        self.seed = seed
        self.task_queue = task_queue
        self.buffer = buffer
        self.max_episode_length = max_episode_length
        self.memory = []
        
        self.model = AQL(self.env,propose_sample=propose_sample, uniform_sample = uniform_sample, action_var = action_var, device = device)
        # self.writer = SummaryWriter(comment="-{}-actor{}".format(env_id, worker_id))

        self.set_all_seeds()
        self.epsilon = epsilon
    def set_all_seeds(self):
        self.env.seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
    def update_weights(self, DQN_state_dict):
        self.model.load_state_dict(DQN_state_dict)
    def record_batch(self):

        self.episode_reward, self.episode_length, episode_idx, actor_idx = 0, 0, 0, 0
        state = self.env.reset()
        self.memory = []
        while True:
            action, a_mu, _ = self.model.act(state, self.epsilon)
            a_mu = a_mu[0]
            next_state, reward, done, _ = self.env.step(a_mu[action])
            self.memory.append((state, action, reward, next_state, done, a_mu))
            
            state = next_state
            self.episode_reward += reward

            self.episode_length += 1
            actor_idx += 1
            if done or self.episode_length >= self.max_episode_length:
                state = self.env.reset()
                # self.writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
                # self.writer.add_scalar("actor/episode_length", episode_length, episode_idx)
                episode_idx += 1
                break
                
    def run(self):
        while True:
            ########## run loop
            task = self.task_queue.get(block=True)
            if task["desc"] == "record_batch":
                # print("start record batch")
                self.record_batch()
                self.buffer.put((self.memory, self.episode_reward, self.episode_length))
                self.task_queue.task_done()
                # print("record batch done")
            elif task["desc"] == "set_pi_weights":
                # print("set weight")
                self.update_weights(task["pi_state_dict"])
                self.task_queue.task_done()
                # print("set weight done")
            elif task["desc"] == "cleanup":
                # print("clean up")
                self.env.close()
                self.task_queue.task_done()
                # print("clean up done")

class BatchRecorder():
    def __init__(self, env_id, env_seed, n_workers, buffer, max_episode_length, writer,
                    propose_sample=100, uniform_sample = 400, action_var = 0.25, device = "cuda"):
        self.env_id = env_id
        # empty batch recorder
        self.n_workers = n_workers
        self.writer = writer
        self.buffer = buffer
        # parallelization
        self.env_seed = env_seed
        self.task_queue = mp.JoinableQueue()
        self.res_queue = mp.Queue()
        self.episode_idx = 0

        self.workers = []
        for i in range(self.n_workers):
            self.workers.append(
                Worker(worker_id=i, env_id=self.env_id, seed=self.env_seed+i, 
                        epsilon= 0.4 ** (1 + i / (self.n_workers - 1) * 7),#0.8 if i < n_workers//3 else 0.05, 
                        max_episode_length=max_episode_length,
                        task_queue=self.task_queue, buffer=self.res_queue,
                        propose_sample=propose_sample, uniform_sample = uniform_sample, 
                        action_var = action_var, device = "cpu"))
        for i, worker in enumerate(self.workers):
            worker.start()


    def record_batch(self):
        task = dict([("desc", "record_batch")])
        total_ep = 0
        for _ in range(self.n_workers):
            self.task_queue.put(task)
        self.task_queue.join()
        for i in range(self.n_workers):
            mem, ep_r, ep_len = self.res_queue.get()
            total_ep+=ep_len
            self.writer.add_scalar("actor/episode_reward", ep_r, self.episode_idx )
            self.writer.add_scalar("actor/episode_length", ep_len, self.episode_idx )
            self.episode_idx += 1
            for (state, action, reward, next_state, done, a_mu) in mem:
                    for _ in range(len(state)):
                        self.buffer.add(state, action, reward, next_state, done, a_mu)

        return total_ep
    def set_worker_weights(self, pi):
        pi.to("cpu")
        task = dict([("desc", "set_pi_weights"),
                     ("pi_state_dict", pi.state_dict())])
        for _ in self.workers:
            self.task_queue.put(task)
        self.task_queue.join()

    def cleanup(self):
        for _ in range(self.n_workers):
            self.task_queue.put(dict([("desc", "cleanup")]))
        for worker in self.workers:
            worker.terminate()

