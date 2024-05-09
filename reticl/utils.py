from typing import Optional
import random
import numpy as np
import torch
from torch import nn

from reticl.constants import Datasets, RLAlgorithm, SamplingMethod, Reward, EncoderModelType, ModelType, Pooling, Init, LRSchedule, DEFAULT_MAX_GEN_TOKENS

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
        self.early_stopping: bool = options_dict.get("early_stopping", False)
        self.model_type: str = options_dict.get("model_type", ModelType.LSTM.value)
        self.model_name: Optional[str] = options_dict.get("model_name", None)
        self.pt_model_name: Optional[str] = options_dict.get("pt_model_name", None)
        self.generator_model: str = options_dict.get("generator_model", "mistralai/Mistral-7B-v0.1")
        self.gpt3_model: str = options_dict.get("gpt3_model", "code-davinci-002")
        self.gen_batch_size: int = options_dict.get("gen_batch_size", 0)
        self.encoder_model_type: str = options_dict.get("encoder_model_type", EncoderModelType.SBERT.value)
        self.encoder_model: str = options_dict.get("encoder_model", None)
        self.encoder_h: int = options_dict.get("encoder_h", 0)
        self.pool: str = options_dict.get("pool", Pooling.MEAN.value)
        self.ft_encoder: bool = options_dict.get("ft_encoder", False)
        self.encoder_lr: Optional[float] = options_dict.get("encoder_lr", None)
        self.soft_prompt_len: int = options_dict.get("soft_prompt_len", 0)
        self.train_size: int = options_dict.get("train_size", 0)
        self.corpus_size: int = options_dict.get("corpus_size", 0)
        self.val_size: int = options_dict.get("val_size", 0)
        self.val_corpus_size: int = options_dict.get("val_corpus_size", 0)
        self.val_every: int = options_dict.get("val_every", 0)
        self.save_best: bool = options_dict.get("save_best", True)
        self.wandb: bool = options_dict.get("wandb", False)
        self.lr: float = options_dict.get("lr", 1e-3)
        self.wd: float = options_dict.get("wd", 1e-2)
        self.adam_eps: float = options_dict.get("adam_eps", 1e-8)
        self.grad_clip: float = options_dict.get("grad_clip", 2.0)
        self.ppo_eps: float = options_dict.get("ppo_eps", 0.2)
        self.tau: float = options_dict.get("tau", 0.01)
        self.replay_buffer_size: int = options_dict.get("replay_buffer_size", 10_000)
        self.updates_per_batch: int = options_dict.get("updates_per_batch", 10)
        self.train_batch_size: int = options_dict.get("train_batch_size", 20)
        self.episodes_before_train: int = options_dict.get("episodes_before_train", 1000)
        self.init: str = options_dict.get("init", Init.ORTHOGONAL.value)
        self.lr_sched: str = options_dict.get("lr_sched", LRSchedule.NONE.value)
        self.epochs: int = options_dict.get("epochs", 50)
        self.inner_epochs: int = options_dict.get("inner_epochs", 4)
        self.batch_size: int = options_dict.get("batch_size", 20)
        self.sub_batch_size: int = options_dict.get("sub_batch_size", 8)
        self.grad_accum_steps: int = options_dict.get("grad_accum_steps", 1)
        self.num_examples: int = options_dict.get("num_examples", 2)
        self.eg_eps: float = options_dict.get("eg_eps", 0.5)
        self.expl_decay_rate: float = options_dict.get("expl_decay_rate", 1.0)
        self.top_k: int = options_dict.get("top_k", 0)
        self.reward: str = options_dict.get("reward", Reward.CONF.value)
        self.int_reward_multi: bool = options_dict.get("int_reward_multi", False)
        self.int_reward_sim: bool = options_dict.get("int_reward_sim", False)
        self.hidden_size: int = options_dict.get("hidden_size", 800)
        self.dropout: float = options_dict.get("dropout", 0.0)
        self.v_coef: float = options_dict.get("v_coef", 0.5)
        self.e_coef: float = options_dict.get("e_coef", 0.0)
        self.cr_coef: float = options_dict.get("cr_coef", 0.5)
        self.anneal_reward: bool = options_dict.get("anneal_reward", False)
        self.lam: float = options_dict.get("lam", 0.95)
        self.gamma: float = options_dict.get("gamma", 0.99)
        self.sep_val_model: bool = options_dict.get("sep_val_model", False)
        self.max_gen_tokens: int = options_dict.get("max_gen_tokens", DEFAULT_MAX_GEN_TOKENS.get(self.dataset, 400))
        self.beam_width: int = options_dict.get("beam_width", 1)
        self.pt_sample_freq: int = options_dict.get("pt_sample_freq", 0)
        self.deterministic: bool = options_dict.get("deterministic", True)
        self.verbose: bool = options_dict.get("verbose", False)

    def as_dict(self):
        return self.__dict__

    def update(self, options_dict: dict):
        self.__dict__.update(options_dict)

def is_pg(options: TrainOptions):
    return options.rl_algo in (RLAlgorithm.REINFORCE.value, RLAlgorithm.RWB.value, RLAlgorithm.AC.value, RLAlgorithm.PPO.value, RLAlgorithm.PPO_SIMPLE.value, RLAlgorithm.DSAC.value)

def max_sbert_len(model_name: str):
    if "all-mpnet-base-v2" in model_name:
        return 384
    if "all-distilroberta-v1" in model_name:
        return 512
    raise Exception(f"Max len not defined for {model_name}!")

def orthogonal_init_(module: nn.Module, gain: float = 1.0):
    for name, param in module.named_parameters():
        if "weight" in name:
            nn.init.orthogonal_(param, gain=gain)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)
