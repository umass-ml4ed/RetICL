from typing import List
import json
import random
import re
import os

from reticl.data_loading.data_types import DataSample
from reticl.utils import TrainOptions

def extract_final_solution(solution: str):
    start_idx = solution.find("\\boxed{")
    macro_len = len("\\boxed{")
    if start_idx == -1:
        start_idx = solution.find("\\fbox{")
        macro_len = len("\\fbox{")
    if start_idx == -1:
        # This isn't super generic but covers the few cases where this actually happens
        start_idx = solution.find("\\boxed ")
        macro_len = len("\\boxed ")
        if start_idx == -1:
            raise Exception("No boxed macro!")
        end_idx = solution.find("$", start_idx)
        final_answer = solution[start_idx + macro_len : end_idx]
        solution_proc = solution[:start_idx] + final_answer + solution[end_idx:]
        return solution_proc, final_answer
    bracket_count = 1
    for idx, char in enumerate(solution[start_idx + macro_len:]):
        if char == "{":
            bracket_count += 1
        elif char == "}":
            bracket_count -= 1
            if bracket_count == 0:
                final_answer = solution[start_idx + macro_len : start_idx + macro_len + idx]
                solution_proc = solution[:start_idx] + final_answer + solution[start_idx + macro_len + idx + 1:]
                return solution_proc, final_answer
    raise Exception("Missing closing bracket!")

def math_load_data(split: str) -> List[dict]:
    samples = []
    split_dir = f"MATH 2/{split}"
    for topic_dir in os.listdir(split_dir):
        if not os.path.isdir(f"{split_dir}/{topic_dir}"):
            continue
        for problem_file in os.listdir(f"{split_dir}/{topic_dir}"):
            if not problem_file.endswith(".json"):
                continue
            with open(f"{split_dir}/{topic_dir}/{problem_file}") as data_file:
                sample = json.load(data_file)
                if sample["level"] not in ("Level 1", "Level 2", "Level 3"):
                    continue
                if "[asy]" in sample["solution"] or "[asy]" in sample["problem"]:
                    continue
                solution, final_answer = extract_final_solution(sample["solution"])
                sample["solution"] = solution
                sample["final_answer"] = final_answer
                samples.append(sample)
    random.Random(221).shuffle(samples)
    return samples

def math_get_data(split: str, options: TrainOptions):
    all_train_data = math_load_data("train")
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
            data = math_load_data(split)
        if options.val_size:
            data = data[:options.val_size]
        corpus = train_data
        if options.val_corpus_size:
            corpus = random.Random(221).sample(corpus, options.val_corpus_size)
    return data, corpus

def math_process_sample(sample: dict) -> DataSample:
    question = re.sub(r"\n{2,}", "\n", sample["problem"])
    answer = re.sub(r"\n{2,}", "\n", sample["solution"]) + "\nFinal Answer: " + sample["final_answer"]
    return {
        "lm_context": f"Problem: {question}\nSolution: ",
        "lm_label": answer,
        "encoder_context": f"Problem: {question}",
        "encoder_label": f"\nSolution: {answer}",
        "meta_data": sample,
    }

def clean_answer(answer: str):
    return answer.replace(",", "").replace(" ", "").replace("dfrac", "frac").replace("^\\circ", "").strip().strip(".").strip("$")

def extract_answer(solution: str, pattern: re.Pattern):
    match = pattern.findall(solution)
    if not match:
        return ""
    return match[-1]

def math_check_correct(src_meta_data: dict, pred_text: str):
    pred_pattern = re.compile(r"Final Answer: (.*)$")
    return clean_answer(src_meta_data["final_answer"]) == clean_answer(extract_answer(pred_text, pred_pattern))

MATH_CONFIG = {
    "get_data": math_get_data,
    "process_sample": math_process_sample,
    "check_correct": math_check_correct,
}
