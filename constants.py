from enum import Enum

class Datasets(Enum):
    TABMWP = "tabmwp"
    GSM8K = "gsm8k"

class EncoderModelType(Enum):
    SBERT = "sbert"
    BERT = "bert"
    GPT2 = "gpt2"

class RLAlgorithm(Enum):
    MCC = "mcc"
    DQN = "dqn"
    REINFORCE = "reinforce"
    RWB = "rwb"
    AC = "ac"
    PPO = "ppo"

class SamplingMethod(Enum):
    RANDOM = "random"
    SIMILARITY = "sim"
    EPSILON_GREEDY = "eg"
    SOFTMAX = "softmax"

class Reward(Enum):
    PPL = "ppl"
    EXACT = "exact"
    EXACT_AND_BLEU = "exact_and_bleu"

MODEL_TO_EMB_SIZE = {
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
}

OPTION_INDS = ["A", "B", "C", "D", "E", "F"] # As defined in PromptPG code
