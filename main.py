import torch
import numpy as np
from tensorboardX import SummaryWriter
from model import DuelingDQN
import utils
from memory import BatchStorage
import _pickle as pickle
from batchrecorder import BatchRecorder
import gym
from multiprocessing import Process
from memory import CustomPrioritizedReplayBuffer
class train_DQN():
    def __init__(self,env_id, seed = 0, lr = 1e-5, n_step = 3, gamma = 0.99, n_workers=2,
                    max_norm = 40, target_update_interval=2500, save_interval = 5000, batch_size = 64,
                    buffer_size = 1e6, prior_alpha = 0.6, prior_beta = 0.4,
                    publish_param_interval = 25, max_step = 1e5):
        self.env = gym.make(env_id)
        self.seed = seed
        self.lr = lr
        self.n_step = n_step
        self.gamma = gamma
        self.max_norm = max_norm
        self.target_update_interval = target_update_interval
        self.save_interval = save_interval
        self.publish_param_interval = publish_param_interval
        self.batch_size = batch_size
        self.prior_beta = prior_beta
        self.max_step = max_step


        self.buffer = CustomPrioritizedReplayBuffer(size=buffer_size,alpha=prior_alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DuelingDQN(self.env).to(self.device)
        self.tgt_model = DuelingDQN(self.env).to(self.device)
        self.tgt_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr, alpha=0.95, eps=1.5e-7, centered=True)

        self.batch_recorder = BatchRecorder(env_id=env_id, env_seed=seed, n_workers=n_workers, buffer= self.buffer,
                n_steps=n_step, gamma=gamma, max_episode_length=50000)
        self.writer = SummaryWriter(comment="-{}-learner".format(self.env.unwrapped.spec.id))
    def train(self):
        utils.set_global_seeds(self.seed, use_torch=True)
        
        learn_idx = 0
        while True:
            batch, idxes = self.buffer.sample(self.batch_size, self.prior_beta)
            loss, prios = utils.compute_loss(self.model, self.tgt_model, batch, self.n_step, self.gamma)
            grad_norm = utils.update_parameters(loss, self.model, self.optimizer, self.max_norm)
            
            self.buffer.update_priorities(idxes, prios)
            
            batch, idxes, prios = None, None, None
            learn_idx += 1

            self.writer.add_scalar("learner/loss", loss, learn_idx)
            self.writer.add_scalar("learner/grad_norm", grad_norm, learn_idx)

            if learn_idx % self.target_update_interval == 0:
                print("Updating Target Network..")
                self.tgt_model.load_state_dict(self.model.state_dict())
            if learn_idx % self.save_interval == 0:
                print("Saving Model..")
                torch.save(self.model.state_dict(), "model{}.pth".format(learn_idx))
            if learn_idx % self.publish_param_interval == 0:
                self.batch_recorder.set_worker_weights(self.model.state_dict())
            if learn_idx >= self.max_step:
                torch.save(self.model.state_dict(), "model{}.pth".format(learn_idx))
                self.batch_recorder.cleanup()
                break

    def sampling_data(self):
        self.batch_recorder.record_batch()

        
if __name__ == "__main__":
    test = train_DQN(env_id="MountainCar-v0")
    procs = [
        Process(target=test.sampling_data()),
        Process(target=test.train())
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()