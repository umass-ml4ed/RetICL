from typing import List
import json
import random
import re

from reticl.data_loading.data_types import DataSample
from reticl.utils import TrainOptions

def qasc_load_data(split: str) -> List[dict]:
    with open(f"QASC_Dataset/{split}.jsonl") as data_file:
        samples = [json.loads(line) for line in data_file]
        random.Random(221).shuffle(samples)
        return samples

def qasc_get_data(split: str, options: TrainOptions):
    all_train_data = qasc_load_data("train")
    train_data, val_data = all_train_data[:-1000], all_train_data[-1000:]
    if split == "train":
        # Get training samples and corpus from train set
        train_size = options.train_size or len(train_data)
        corpus_size = options.corpus_size or len(train_data) - train_size
        data = train_data[:train_size]
        corpus = train_data[train_size : train_size + corpus_size]
    else:
        # Get evaluation samples from split and corpus from train set
        if split == "dev":
            data = val_data
        elif split == "test":
            data = qasc_load_data("dev")
        if options.val_size:
            data = data[:options.val_size]
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def qasc_process_sample(sample: dict) -> DataSample:
    question = sample["formatted_question"]
    solution = " ".join([sample["fact1"], sample["fact2"], sample["combinedfact"]])
    # solution = sample["combinedfact"]
    answer_idx = ord(sample["answerKey"]) - ord("A")
    answer = sample["question"]["choices"][answer_idx]["text"]
    return {
        "lm_context": f"Question: {question}\nSolution:",
        "lm_label": f" {solution}\nFinal Answer: {answer}",
        "encoder_context": f"Question: {question}",
        "encoder_label": f"\nSolution: {solution}\nFinal Answer: {answer}",
        "meta_data": {**sample, "answer": answer},
    }

def extract_answer(solution: str):
    match = re.findall(r"Final Answer: (.*)$", solution)
    if not match:
        return ""
    return match[-1]

def clean_answer(answer: str):
    return answer.strip().replace(" ", "").strip(".").lower()

def qasc_check_correct(src_meta_data: dict, pred_text: str):
    return clean_answer(src_meta_data["answer"]) == clean_answer(extract_answer(pred_text))

def qasc_complexity_metric(sample: DataSample):
    return sample["lm_label"].count(" ")

QASC_CONFIG = {
    "get_data": qasc_get_data,
    "process_sample": qasc_process_sample,
    "check_correct": qasc_check_correct,
    "complexity_metric": qasc_complexity_metric,
}
