from typing import List
import json
import random
import re

from reticl.data_loading.data_types import DataSample
from reticl.utils import TrainOptions

def gsm8k_load_data(split: str) -> List[dict]:
    with open(f"grade-school-math/grade_school_math/data/{split}.jsonl") as data_file:
        samples = [json.loads(line) for line in data_file]
        random.Random(221).shuffle(samples)
        return samples

def gsm8k_get_data(split: str, options: TrainOptions):
    all_train_data = gsm8k_load_data("train")
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
        else:
            data = gsm8k_load_data(split)
        if options.val_size:
            data = data[:options.val_size]
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def gsm8k_process_sample(sample: dict) -> DataSample:
    # question = sample["question"].replace("\n", "\\n")
    # answer = remove_calc_annotations(sample["answer"]).replace("\n", "\\n")
    question = sample["question"]
    answer = remove_calc_annotations(sample["answer"])
    answer = answer.replace("####", "Final Answer:")
    answer_lines = answer.split("\n")
    return {
        "lm_context": f"Problem: {question}\nSolution:",
        "lm_label": f" {answer}",
        # "encoder_context": f"Question: {question}\nAnswer: ",
        # "encoder_label": answer,
        "encoder_context": f"Problem: {question}",
        "encoder_label": f"\nSolution: {answer}",
        # "encoder_context": f"Problem: {question}",
        # "encoder_label": f"Full Solution: [SEP]{'[SEP]'.join(answer_lines)}",
        # "encoder_label": f"Full Solution: {answer}",
        "meta_data": sample,
    }

def remove_calc_annotations(solution: str):
    result = ""
    start_idx = solution.find("<<")
    end_idx = 0
    while start_idx != -1:
        result += solution[end_idx : start_idx]
        end_idx = solution.find(">>", start_idx) + 2
        start_idx = solution.find("<<", end_idx)
    result += solution[end_idx:]
    return result

def extract_answer(solution: str, pattern: re.Pattern):
    chars_to_remove = re.compile(r"[,\$%]")
    match = pattern.findall(chars_to_remove.sub("", solution))
    if not match:
        return ""
    return match[-1].strip().strip(".").replace(".00", "")

def gsm8k_check_correct(src_meta_data: dict, pred_text: str):
    src_pattern = re.compile(r"#### (.*)$")
    pred_pattern = re.compile(r"Final Answer: ([\d\.]+)")
    return extract_answer(src_meta_data["answer"], src_pattern) == extract_answer(pred_text, pred_pattern)

def gsm8k_complexity_metric(sample: DataSample):
    return sample["lm_label"].count("\n")

GSM8K_CONFIG = {
    "get_data": gsm8k_get_data,
    "process_sample": gsm8k_process_sample,
    "check_correct": gsm8k_check_correct,
    "complexity_metric": gsm8k_complexity_metric,
}
