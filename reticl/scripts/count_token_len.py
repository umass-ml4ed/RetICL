from transformers import AutoTokenizer
import torch

from reticl.utils import TrainOptions
from reticl.datasets.tabmwp import tabmwp_get_data
from reticl.datasets.gsm8k import gsm8k_get_data
from reticl.datasets.math import math_get_data
from reticl.datasets.qasc import qasc_get_data
from reticl.datasets.mtop import mtop_get_data
from reticl.datasets.svamp import svamp_get_data
from reticl.datasets.ecqa import ecqa_get_data

def get_sol(dataset, sample):
    if dataset == "qasc":
        return " ".join([sample["fact1"], sample["fact2"], sample["combinedfact"]])
    elif dataset == "mtop":
        return sample["decoupled"]
    elif dataset == "math":
        return sample["solution"]
    elif dataset == "svamp":
        return sample["Equation"] + " " + str(sample["Answer"])
    elif dataset == "ecqa":
        return sample["taskB"] + sample["q_ans"]
    else:
        return sample["answer"]

def count_token_len():
    dataset = "ecqa"
    get_data = ecqa_get_data
    options = TrainOptions({"train_size": 0, "corpus_size": 0})
    train_data = get_data("train", options)[0]
    val_data = get_data("dev", options)[0]
    test_data = get_data("test", options)[0]
    print(f"Num samples: train {len(train_data)}, val {len(val_data)}, test {len(test_data)}")
    data = train_data + val_data + test_data
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    num_chars = torch.tensor([len(get_sol(dataset, sample)) for sample in data])
    num_tokens = torch.tensor([len(tokenizer(get_sol(dataset, sample)).input_ids) for sample in data])
    print("num chars", num_chars.topk(10))
    print("num tokens", num_tokens.topk(10))

if __name__ == "__main__":
    count_token_len()
