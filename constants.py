from enum import Enum

class Datasets(Enum):
    TABMWP = "tabmwp"
    GSM8K = "gsm8k"

class SamplingMethod(Enum):
    RANDOM = "random"
    SIMILARITY = "sim"
    MCC = "mcc"
    PG = "pg"
    RWB = "rwb"
    AC = "ac"
    PPO = "ppo"

class Reward(Enum):
    PPL = "ppl"
    EXACT = "exact"
    EXACT_AND_BLEU = "exact_and_bleu"

MODEL_TO_EMB_SIZE = {
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
}

OPTION_INDS = ["A", "B", "C", "D", "E", "F"] # As defined in PromptPG code
