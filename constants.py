from enum import Enum

class SamplingMethod(Enum):
    RANDOM = "random"
    SIMILARITY = "sim"
    MCC = "mcc"
    PG = "pg"

class Reward(Enum):
    PPL = "ppl"
    EXACT = "exact"

MODEL_TO_EMB_SIZE = {
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
}

OPTION_INDS = ["A", "B", "C", "D", "E", "F"] # As defined in PromptPG code
