from typing import List
import random
import re
import pandas as pd

from reticl.data_loading.data_types import DataSample
from reticl.utils import TrainOptions

def mtop_load_data(split: str) -> List[dict]:
    data = pd.read_csv(
        f"mtop/en/{split}.txt", sep="\t", header=None,
        names=["id", "intent", "slot", "utterance", "domain", "locale", "decoupled", "tokens"]
    ).to_dict("records")
    random.Random(221).shuffle(data)
    return data

def mtop_get_data(split: str, options: TrainOptions):
    train_data = mtop_load_data("train")
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
            data = mtop_load_data("eval")
        else:
            data = mtop_load_data(split)
        if options.val_size:
            data = data[:options.val_size]
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def mtop_process_sample(sample: dict) -> DataSample:
    utterance = sample["utterance"]
    representation = sample["decoupled"]
    return {
        "lm_context": f"Utterance: {utterance}\nRepresentation:",
        "lm_label": f" {representation}",
        "encoder_context": f"Utterance: {utterance}",
        "encoder_label": f"\nRepresentation: {representation}",
        "meta_data": sample,
    }

def extract_answer(solution: str, pattern: re.Pattern):
    chars_to_remove = re.compile(r"[,\$%]")
    match = pattern.findall(chars_to_remove.sub("", solution))
    if not match:
        return ""
    return match[-1].strip().strip(".").replace(".00", "")

def mtop_check_correct(src_meta_data: dict, pred_text: str):
    return src_meta_data["decoupled"].strip() == pred_text.strip()

def mtop_complexity_metric(sample: DataSample):
    return sample["lm_label"].count(" ")

MTOP_CONFIG = {
    "get_data": mtop_get_data,
    "process_sample": mtop_process_sample,
    "check_correct": mtop_check_correct,
    "complexity_metric": mtop_complexity_metric,
}
