from typing import List, TypedDict, Tuple
import torch
import numpy as np

from reticl.data_loading.reticl_dataset import ICLSample
from reticl.utils import TrainOptions

class Episode(TypedDict):
    sample: ICLSample
    rewards: torch.Tensor

class ReplayBuffer:
    def __init__(self, options: TrainOptions):
        self.buffer_size = options.replay_buffer_size
        self.buffer: List[Episode] = []
        self.pos = 0

    def add(self, batch: List[ICLSample], rewards: torch.Tensor):
        for sample, reward in zip(batch, rewards):
            episode = {"sample": sample, "rewards": reward}
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(episode)
            else:
                self.buffer[self.pos] = episode
            self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, n: int) -> Tuple[List[ICLSample], torch.Tensor]:
        # Assign weights so more recent samples are more likely
        episodes: List[Episode] = np.random.choice(self.buffer, size=n)
        return [ep["sample"] for ep in episodes], torch.stack([ep["rewards"] for ep in episodes])

    def __len__(self):
        return len(self.buffer)
