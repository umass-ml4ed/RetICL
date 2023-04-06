from typing import Optional
import random
import numpy as np
import torch

from constants import Datasets, RLAlgorithm, SamplingMethod, Reward, EncoderModelType, ModelType, Init

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
        if self.rl_algo is not None and self.sm in (SamplingMethod.RANDOM.value, SamplingMethod.SIMILARITY.value):
            raise Exception("RL algorithm not used with random or similarity sampling!")
        self.model_type: str = options_dict.get("model_type", ModelType.LSTM.value)
        self.model_name: Optional[str] = options_dict.get("model_name", None)
        self.generator_model: str = options_dict.get("generator_model", "gpt3")
        self.gpt3_model: str = options_dict.get("gpt3_model", "code-davinci-002")
        self.gen_batch_size: int = options_dict.get("gen_batch_size", 0)
        self.encoder_model_type: str = options_dict.get("encoder_model_type", EncoderModelType.SBERT.value)
        self.encoder_model: str = options_dict.get("encoder_model", None)
        self.soft_prompt_len: int = options_dict.get("soft_prompt_len", 0)
        self.train_size: int = options_dict.get("train_size", 1000)
        self.corpus_size: int = options_dict.get("corpus_size", 0)
        self.val_size: int = options_dict.get("val_size", 0)
        self.val_corpus_size: int = options_dict.get("val_corpus_size", 0)
        self.save_best: bool = options_dict.get("save_best", True)
        self.wandb: bool = options_dict.get("wandb", False)
        self.lr: float = options_dict.get("lr", 1e-3)
        self.wd: float = options_dict.get("wd", 1e-2)
        self.grad_clip: float = options_dict.get("grad_clip", 2.0)
        self.ppo_eps: float = options_dict.get("ppo_eps", 0.1)
        self.init: str = options_dict.get("init", Init.ORTHOGONAL.value)
        self.epochs: int = options_dict.get("epochs", 20)
        self.batch_size: int = options_dict.get("batch_size", 20)
        self.grad_accum_steps: int = options_dict.get("grad_accum_steps", 1)
        self.num_examples: int = options_dict.get("num_examples", 2)
        self.eg_eps: float = options_dict.get("eg_eps", 0.5)
        self.expl_decay_rate: float = options_dict.get("expl_decay_rate", 1.0)
        self.top_k: int = options_dict.get("top_k", 0)
        self.reward: str = options_dict.get("reward", Reward.EXACT.value)
        self.hidden_size: int = options_dict.get("hidden_size", 800)
        self.dropout: float = options_dict.get("dropout", 0.0)
        self.v_coef: float = options_dict.get("v_coef", 0.5)
        self.e_coef: float = options_dict.get("e_coef", 0.0)
        self.sep_val_model: bool = options_dict.get("sep_val_model", False)
        self.max_gen_tokens: int = options_dict.get("max_gen_tokens", 400)
        self.deterministic: bool = options_dict.get("deterministic", True)

    def as_dict(self):
        return self.__dict__

    def update(self, options_dict: dict):
        self.__dict__.update(options_dict)

def is_pg(options: TrainOptions):
    return options.rl_algo in (RLAlgorithm.REINFORCE.value, RLAlgorithm.RWB.value, RLAlgorithm.AC.value, RLAlgorithm.PPO.value)

def max_sbert_len(model_name: str):
    if "all-mpnet-base-v2" in model_name:
        return 384
    if "all-distilroberta-v1" in model_name:
        return 512
    raise Exception(f"Max len not defined for {model_name}!")
