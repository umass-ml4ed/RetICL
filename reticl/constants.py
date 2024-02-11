from enum import Enum

class Datasets(Enum):
    TABMWP = "tabmwp"
    GSM8K = "gsm8k"
    MATH = "math"
    SVAMP = "svamp"
    QASC = "qasc"
    CQA = "cqa"
    AGNEWS = "agnews"

class ModelType(Enum):
    RNN = "rnn"
    LSTM = "lstm"
    ATTN = "attn"
    IND = "ind"

class EncoderModelType(Enum):
    SBERT = "sbert"
    BERT = "bert"
    GPT2 = "gpt2"

class RLAlgorithm(Enum):
    MCC = "mcc"
    REINFORCE = "reinforce"
    RWB = "rwb"
    AC = "ac"
    PPO = "ppo"
    PPO_SIMPLE = "ppo_simple"
    DSAC = "dsac"

class SamplingMethod(Enum):
    RANDOM = "random"
    SIMILARITY = "sim"
    EPSILON_GREEDY = "eg"
    SOFTMAX = "softmax"
    EXHAUSTIVE = "exhaustive"
    COMPLEX = "complex"
    VF = "vf"

class Reward(Enum):
    PPL = "ppl"
    EXACT = "exact"
    EXACT_AND_PPL = "exact_and_ppl"
    EXACT_AND_BLEU = "exact_and_bleu"

class Pooling(Enum):
    MEAN = "mean"
    ATTN = "attn"

class Init(Enum):
    DEFAULT = "default"
    ORTHOGONAL = "ortho"

class LRSchedule(Enum):
    NONE = "none"
    LINEAR = "linear"
    CYCLE = "cycle"

MODEL_TO_EMB_SIZE = {
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
    "all-distilroberta-v1": 768,
}

DEFAULT_MAX_GEN_TOKENS = {
    Datasets.TABMWP.value: 450,
    Datasets.GSM8K.value: 400,
    Datasets.QASC.value: 150,
    Datasets.CQA.value: 100,
    Datasets.AGNEWS.value: 10,
}
