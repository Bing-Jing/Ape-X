import random
import gym
import numpy as np
import _pickle as pickle
import multiprocessing as mp
import torch
from model import DuelingDQN
from memory import BatchStorage
from tensorboardX import SummaryWriter
import itertools


class Worker(mp.Process):
    def __init__(self, worker_id, env_id, seed, epsilon, size, lock,
                n_steps, gamma, send_interval, task_queue, buffer, max_episode_length):
        mp.Process.__init__(self)
        self.worker_id = worker_id
        self.env = gym.make(env_id)
        self.seed = seed
        self.task_queue = task_queue
        self.buffer = buffer
        self.max_episode_length = max_episode_length
        self.send_interval = send_interval
        self.storage = BatchStorage(n_steps, gamma)
        self.size = size
        self.memory = []
        
        self.model = DuelingDQN(self.env)
        self.writer = SummaryWriter(comment="-{}-actor{}".format(env_id, worker_id))
        self.lock = lock

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

        episode_reward, episode_length, episode_idx, actor_idx = 0, 0, 0, 0
        state = self.env.reset()
        self.storage.reset()
        self.memory = []
        # while actor_idx < self.size:
        while True:
            action, q_values = self.model.act(torch.FloatTensor(np.array(state)), self.epsilon)
            next_state, reward, done, _ = self.env.step(action)
            
            
            self.storage.add(state, reward, action, done, q_values)

            state = next_state
            episode_reward += reward
            episode_length += 1
            actor_idx += 1
            if done or episode_length >= self.max_episode_length:
                state = self.env.reset()
                self.writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
                self.writer.add_scalar("actor/episode_length", episode_length, episode_idx)
                episode_reward = 0
                episode_length = 0
                episode_idx += 1


            if done or len(self.storage) == self.send_interval:
                batch, prios = self.storage.make_batch()
                self.memory.append((*batch, prios))
                # for i in range(len(prios)):
                #     self.buffer.add(batch[0][i],batch[1][i],batch[2][i],batch[3][i],batch[4][i],prios[i])

                batch, prios = None, None
                self.storage.reset()
            if done:
                break
    def run(self):
        while True:
            ########## run loop
            task = self.task_queue.get(block=True)
            if task["desc"] == "record_batch":
                # print("start record batch")
                self.record_batch()
                self.buffer.put(self.memory)
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
    def __init__(self, env_id, env_seed, n_workers, buffer,
                n_steps, gamma, max_episode_length):
        self.env_id = env_id
        lock = mp.Lock()
        # empty batch recorder
        self.n_workers = n_workers

        self.buffer = buffer
        # parallelization
        self.env_seed = env_seed
        self.task_queue = mp.JoinableQueue()
        self.res_queue = mp.Queue()
        self.size = 1e5

        self.worker_batch_sizes = [self.size // self.n_workers] * self.n_workers

        self.workers = []
        for i in range(self.n_workers):
            self.workers.append(
                Worker(worker_id=i, env_id=self.env_id, seed=self.env_seed+i, 
                        epsilon= 0.4 ** (1 + i / (n_workers - 1) * 7), n_steps=n_steps,
                        gamma=gamma, send_interval=50, size = self.worker_batch_sizes[i], max_episode_length=max_episode_length,
                        task_queue=self.task_queue, buffer=self.res_queue, lock = lock))
        for i, worker in enumerate(self.workers):
            worker.start()


    def record_batch(self):
        task = dict([("desc", "record_batch")])
        for _ in range(self.n_workers):
            self.task_queue.put(task)
        self.task_queue.join()
        for i in range(self.n_workers):
            mem = self.res_queue.get()
            for sample in mem:
                for i in range(len(sample[0])):
                    self.buffer.add(sample[0][i],sample[1][i],sample[2][i],sample[3][i],sample[4][i],sample[5][i])


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

