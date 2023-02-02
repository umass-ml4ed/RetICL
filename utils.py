from typing import Optional
import random
import numpy as np
import torch

from constants import Datasets, RLAlgorithm, SamplingMethod, Reward, EncoderModelType

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
        self.dataset: str = options_dict.get("dataset", Datasets.GSM8K.value)
        self.rl_algo: Optional[str] = options_dict.get("rl_algo", None)
        self.sm: str = options_dict.get("sm", SamplingMethod.SOFTMAX.value)
        if self.rl_algo is None and self.sm in (SamplingMethod.EPSILON_GREEDY.value, SamplingMethod.SOFTMAX.value):
            raise Exception("Must define RL algorithm if using epsilon-greedy or softmax sampling!")
        if self.rl_algo is not None and self.sm in (SamplingMethod.RANDOM.value, SamplingMethod.SIMILARITY.value):
            raise Exception("RL algorithm not used with random or similarity sampling!")
        self.model_name: Optional[str] = options_dict.get("model_name", None)
        self.generator_model: str = options_dict.get("generator_model", "gpt3") # "EleutherAI/gpt-j-6B"
        self.gpt3_model: str = options_dict.get("gpt3_model", "code-davinci-002")
        self.encoder_model_type: str = options_dict.get("encoder_model_type", EncoderModelType.SBERT.value)
        self.encoder_model: str = options_dict.get("encoder_model", None)
        self.train_size: int = options_dict.get("train_size", 1000)
        self.corpus_size: int = options_dict.get("corpus_size", 0)
        self.wandb: bool = options_dict.get("wandb", False)
        self.baseline: bool = options_dict.get("baseline", False)
        self.lr: float = options_dict.get("lr", 1e-3)
        self.wd: float = options_dict.get("wd", 1e-2)
        self.epochs: int = options_dict.get("epochs", 20)
        self.batch_size: int = options_dict.get("batch_size", 20)
        self.grad_accum_steps: int = options_dict.get("grad_accum_steps", 1)
        self.num_examples: int = options_dict.get("num_examples", 2)
        self.epsilon: float = options_dict.get("epsilon", 0.5)
        self.epsilon_decay: float = options_dict.get("epsilon_decay", 0.9)
        self.top_k: int = options_dict.get("top_k", 0)
        self.reward: str = options_dict.get("reward", Reward.EXACT.value)
        self.hidden_size: int = options_dict.get("hidden_size", 100)
        self.dropout: float = options_dict.get("dropout", 0.0)
        self.v_coef: float = options_dict.get("v_coef", 0.5)
        self.e_coef: float = options_dict.get("e_coef", 0.0)

    def as_dict(self):
        return self.__dict__

    def update(self, options_dict: dict):
        self.__dict__.update(options_dict)

def is_pg(options: TrainOptions):
    return options.rl_algo in (RLAlgorithm.REINFORCE.value, RLAlgorithm.RWB.value, RLAlgorithm.AC.value, RLAlgorithm.PPO.value)
