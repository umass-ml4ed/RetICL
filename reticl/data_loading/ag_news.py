from typing import List
import json
import random
import re
from datasets import load_dataset

from reticl.data_loading.data_types import DataSample
from reticl.utils import TrainOptions

TOPICS = ["World", "Sports", "Business", "Sci/Tech"]

def agnews_load_data(split: str) -> List[dict]:
    samples = load_dataset("ag_news", split=split)
    samples = [sample for sample in samples] # Convert from Dataset to list
    random.Random(221).shuffle(samples)
    return samples

def agnews_get_data(split: str, options: TrainOptions):
    all_train_data = agnews_load_data("train")
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
        else:
            data = agnews_load_data(split)
        if options.val_size:
            data = data[:options.val_size]
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def get_label(sample: dict) -> str:
    return TOPICS[sample["label"]]

def agnews_process_sample(sample: dict) -> DataSample:
    text = sample["text"]
    topics = ", ".join(TOPICS)
    label = get_label(sample)
    return {
        "lm_context": f"News Snippet: {text}\nPossible Topics: {topics}\nLabel:",
        "lm_label": f" {label}",
        "encoder_context": f"News Snippet: {text}\nPossible Topics: {topics}",
        "encoder_label": f"\nLabel: {label}",
        "meta_data": sample,
    }

def agnews_check_correct(src_meta_data: dict, pred_text: str):
    return get_label(src_meta_data) == pred_text.strip()

AGNEWS_CONFIG = {
    "get_data": agnews_get_data,
    "process_sample": agnews_process_sample,
    "check_correct": agnews_check_correct,
}
