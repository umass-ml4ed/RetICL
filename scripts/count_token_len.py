from transformers import GPT2TokenizerFast
import torch

from utils import TrainOptions
from data_loading.tabmwp import tabmwp_get_data
from data_loading.gsm8k import gsm8k_get_data
from data_loading.math import math_get_data
from data_loading.qasc import qasc_get_data

def get_sol(dataset, sample):
    if dataset == "qasc":
        return " ".join([sample["fact1"], sample["fact2"], sample["combinedfact"]])
    else:
        return sample["answer"]

def count_token_len():
    dataset = "qasc"
    get_data = qasc_get_data
    options = TrainOptions({"train_size": 0, "corpus_size": 0})
    data = get_data("train", options)[0] + get_data("dev", options)[0] + get_data("test", options)[0]
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    num_chars = torch.tensor([len(get_sol(dataset, sample)) for sample in data])
    num_tokens = torch.tensor([len(tokenizer(get_sol(dataset, sample)).input_ids) for sample in data])
    print("num chars", num_chars.topk(10))
    print("num tokens", num_tokens.topk(10))

if __name__ == "__main__":
    count_token_len()
