from typing import List
import json
import random
import re

from data_loading.data_types import DataSample
from utils import TrainOptions

def svamp_load_data() -> List[dict]:
    with open("SVAMP/SVAMP.json") as data_file:
        samples = json.load(data_file)
        random.Random(221).shuffle(samples)
        return samples

def svamp_get_data(split: str, options: TrainOptions):
    all_data = svamp_load_data()
    train_data, val_data, test_data = all_data[:700], all_data[700:800], all_data[800:]
    if split == "train":
        # Get training samples and corpus from train set
        train_size = options.train_size or len(train_data)
        data = train_data[:train_size]
        if options.corpus_size:
            corpus = train_data[train_size : train_size + options.corpus_size]
        else:
            corpus = None
    else:
        # Get evaluation samples from split and corpus from train set
        if split == "dev":
            data = val_data
        elif split == "test":
            data = test_data
        if options.val_size:
            data = data[:options.val_size]
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def svamp_process_sample(sample: dict) -> DataSample:
    body = sample["Body"]
    if not body.endswith("."):
        body += "."
    question = body + " " + sample["Question"]
    equation = sample["Equation"]
    answer = sample["Answer"]
    return {
        "lm_context": f"Problem: {question}\nEquation:",
        "lm_label": f" {equation}\nFinal Answer: {answer}",
        "encoder_context": f"Problem: {question}",
        "encoder_label": f"\nEquation: {equation}\nFinal Answer: {answer}",
        "meta_data": sample,
    }

def svamp_check_correct(src_meta_data: dict, pred_text: str):
    pred_pattern = re.compile(r"Final Answer: ([\d\.]+)")
    match = pred_pattern.findall(pred_text)
    if not match:
        return False
    pred_answer = match[-1].strip().strip(".")
    return str(src_meta_data["Answer"]) == pred_answer
