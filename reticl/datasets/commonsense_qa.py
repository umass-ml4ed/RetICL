from typing import List
import json
import random
import re

from reticl.data_loading.data_types import DataSample
from reticl.utils import TrainOptions

def cqa_load_data(split: str) -> List[dict]:
    with open(f"commonsense_qa/{split}_rand_split.jsonl") as data_file:
        samples = [json.loads(line) for line in data_file]
        random.Random(221).shuffle(samples)
        return samples

def cqa_get_data(split: str, options: TrainOptions):
    all_train_data = cqa_load_data("train")
    train_data, val_data = all_train_data[:-1000], all_train_data[-1000:]
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
            data = val_data
        elif split == "test":
            data = cqa_load_data("dev")
        if options.val_size:
            data = data[:options.val_size]
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def get_answer(sample: dict) -> str:
    return [choice["text"] for choice in sample["question"]["choices"] if choice["label"] == sample["answerKey"]][0]

def cqa_process_sample(sample: dict) -> DataSample:
    question = sample["question"]["stem"]
    options = " ".join([choice["label"] + ") " + choice["text"] for choice in sample["question"]["choices"]])
    answer = get_answer(sample)
    return {
        "lm_context": f"Question: {question}\nOptions: {options}\nAnswer:",
        "lm_label": f" {answer}",
        "encoder_context": f"Question: {question}\nOptions: {options}",
        "encoder_label": f"\nAnswer: {answer}",
        "meta_data": sample,
    }

def cqa_check_correct(src_meta_data: dict, pred_text: str):
    return get_answer(src_meta_data) == pred_text.strip()

CQA_CONFIG = {
    "get_data": cqa_get_data,
    "process_sample": cqa_process_sample,
    "check_correct": cqa_check_correct,
}
