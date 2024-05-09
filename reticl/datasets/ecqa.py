from typing import List
import random
import pandas as pd

from reticl.data_loading.data_types import DataSample
from reticl.utils import TrainOptions

def ecqa_load_data(split: str) -> List[dict]:
    df = pd.read_csv(f"ECQA-Dataset/cqa_data_{split}.csv")
    df = df.sample(frac=1, random_state=221)
    return df.to_dict("records")

def ecqa_get_data(split: str, options: TrainOptions):
    train_data = ecqa_load_data("train")
    if split == "train":
        # Get training samples and corpus from train set
        if not options.train_size and not options.corpus_size:
            data = train_data
            corpus = None
        else:
            train_size = options.train_size or len(train_data) - options.corpus_size
            corpus_size = options.corpus_size or len(train_data) - train_size
            data = train_data[:train_size]
            corpus = train_data[train_size : train_size + corpus_size]
    else:
        # Get evaluation samples from split and corpus from train set
        if split == "dev":
            data = ecqa_load_data("val")
        elif split == "test":
            data = ecqa_load_data(split)
        if options.val_size:
            data = data[:options.val_size]
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def ecqa_process_sample(sample: dict) -> DataSample:
    question = sample["q_text"]
    options = ", ".join([
        letter + ") " + sample[f"q_op{idx}"]
        for letter, idx in zip(["A", "B", "C", "D", "E"], range(1, 6))
    ])
    reasoning = sample["taskB"]
    answer = sample["q_ans"]
    return {
        "lm_context": f"Question: {question}\nOptions: {options}\nReasoning:",
        "lm_label": f" {reasoning}\nAnswer: {answer}",
        "encoder_context": f"Question: {question}\nOptions: {options}",
        "encoder_label": f"\nReasoning: {reasoning}\nAnswer: {answer}",
        "meta_data": sample,
    }

def ecqa_check_correct(src_meta_data: dict, pred_text: str):
    split = pred_text.split("Answer: ")
    return len(split) == 2 and src_meta_data["q_ans"] == split[1].strip()

ECQA_CONFIG = {
    "get_data": ecqa_get_data,
    "process_sample": ecqa_process_sample,
    "check_correct": ecqa_check_correct,
}
