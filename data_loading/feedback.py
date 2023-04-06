from typing import List
import json
import random
import re
import nltk

from data_loading.data_types import DataSample
from utils import TrainOptions

def eedi_load_data() -> List[dict]:
    with open("eedi_23_data/ed_data_with_explanation_and_proportion.json") as data_file:
        samples = json.load(data_file)
        random.Random(221).shuffle(samples)
        return samples

def expand_options(data: List[dict]):
    result = []
    for sample in data:
        for distractor in sample["distractors"]:
            if "explantion" in distractor:
                distractor["explanation"] = distractor["explantion"]
                del distractor["explantion"]
            result.append({
                "question": sample["question"],
                **distractor,
            })
    return result

def eedi_get_data(split: str, options: TrainOptions):
    all_data = eedi_load_data()
    train_size = int(.6 * len(all_data))
    test_size = int(.2 * len(all_data))
    train_data, val_data, test_data = all_data[:train_size], all_data[train_size : -test_size], all_data[-test_size:]
    if split == "train":
        # Get training samples and corpus from train set
        train_size = options.train_size or len(train_data)
        data = expand_options(train_data[:train_size])
        corpus_size = options.corpus_size or len(train_data) - train_size
        corpus = expand_options(train_data[train_size : train_size + corpus_size])
    else:
        # Get evaluation samples from split and corpus from train set
        if split == "dev":
            data = val_data
        elif split == "test":
            data = test_data
        if options.val_size:
            data = data[:options.val_size]
        data = expand_options(data)
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
        corpus = expand_options(corpus)
    return data, corpus

def eedi_process_sample(sample: dict) -> DataSample:
    question = sample["question"]
    answer = sample["option"]
    explanation = sample["explanation"]
    return {
        "lm_context": f"Problem: {question}\nIncorrect Answer: {answer}\nFeedback:",
        "lm_label": f" {explanation}",
        "encoder_context": f"Problem: {question}\nIncorrect Answer: {answer}",
        "encoder_label": f"\nFeedback: {explanation}",
        "meta_data": sample,
    }

def eedi_check_correct(src_meta_data: dict, pred_text: str):
    pred_answer = pred_text.strip()
    return nltk.translate.bleu([src_meta_data["explanation"]], pred_answer)
