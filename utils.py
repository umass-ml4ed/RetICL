from typing import Optional
import random
from functools import lru_cache
import numpy as np
import torch

from constants import SamplingMethod, Reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

class TrainOptions:
    def __init__(self, options_dict: dict):
        self.method: str = options_dict.get("method", SamplingMethod.MCC.value)
        self.model_name: Optional[str] = options_dict.get("model_name", None)
        self.generator_model: str = options_dict.get("generator_model", "gpt3") # "EleutherAI/gpt-j-6B"
        self.gpt3_model: str = options_dict.get("gpt3_model", "code-davinci-002")
        self.encoder_model: str = options_dict.get("encoder_model", "all-mpnet-base-v2")
        self.train_size: int = options_dict.get("train_size", 1000)
        self.corpus_size: int = options_dict.get("corpus_size", 0)
        self.wandb: bool = options_dict.get("wandb", False)
        self.lr: float = options_dict.get("lr", 1e-3)
        self.epochs: int = options_dict.get("epochs", 20)
        self.batch_size: int = options_dict.get("batch_size", 20)
        self.num_examples: int = options_dict.get("num_examples", 2)
        self.epsilon: float = options_dict.get("epsilon", 0.5)
        self.epsilon_decay: float = options_dict.get("epsilon_decay", 0.9)
        self.top_k: int = options_dict.get("top_k", 0)
        self.reward: str = options_dict.get("reward", Reward.EXACT.value)
        self.hidden_size: int = options_dict.get("hidden_size", 100)

    def as_dict(self):
        return self.__dict__

    def update(self, options_dict: dict):
        self.__dict__.update(options_dict)
