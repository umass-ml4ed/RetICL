from typing import List
import json
import random
import re

from data_loading.data_types import DataSample
from promptPG.base_prompt import get_table_text, get_question_text, get_answer, get_solution_text
from promptPG.utilities import normalize_answer
from constants import OPTION_INDS
from utils import TrainOptions

def tabmwp_load_data(split: str) -> List[dict]:
    with open(f"tabmwp/problems_{split}.json", encoding="utf-8") as data_file:
        samples = list(json.load(data_file).values())
        random.Random(221).shuffle(samples)
        return samples

def tabmwp_get_data(split: str, options: TrainOptions):
    if split == "train":
        # Get training samples and corpus from train set
        train_data = tabmwp_load_data("train")
        train_size = options.train_size or len(train_data)
        corpus_size = options.corpus_size or len(train_data) - train_size
        data = train_data[:train_size]
        corpus = train_data[train_size : train_size + corpus_size]
    else:
        # Get evaluation samples from split and corpus from train set
        if split == "dev":
            data = tabmwp_load_data("dev1k")[:200]
        elif split == "dev100":
            data = tabmwp_load_data("dev")[:100]
        else:
            data = tabmwp_load_data(split)
        corpus = tabmwp_load_data("train")
        if split.startswith("test") and options.corpus_size != 0:
            corpus = random.Random(221).sample(corpus, options.corpus_size)
    return data, corpus

def tabmwp_process_sample(sample: dict) -> DataSample:
    # Prompt structure follows from PromptPG paper
    table = get_table_text(sample)
    question = get_question_text(sample, OPTION_INDS)
    answer = get_answer(sample)
    solution = get_solution_text(sample)
    return {
        "lm_context": f"Table: {table}\nProblem: {question}\nSolution:",
        "lm_label": f" {solution}\nFinal Answer: {answer}",
        "encoder_context": f"Table: {table}\nProblem: {question}",
        "encoder_label": f"\nSolution: {solution}\nFinal Answer: {answer}",
        # "lm_context": f"Table: {table}\nQuestion: {question}\nAnswer:",
        # "lm_label": f" {solution} The answer is {answer}.",
        # "encoder_context": f"Table: {table}\nQuestion: {question}",
        # "encoder_label": f"Answer: {solution} The answer is {answer}.",
        "meta_data": sample,
    }

def extract_prediction(output: str, options: List[str]):
    # $\\frac{16}{95}$ -> 16/95
    output = re.sub(r"\$?\\frac\{([\d\.\,\-]+)\}\{([\d\.\,]+)\}\$?", r"\1/\2", output)
    output = re.sub(r"(?<![AP]\.M)\.$", "", output)
    output = re.sub(r"(?<=\d)[\=](?=[\-\$\d])", " = ", output)
    output = re.sub(r"\u2212", "-", output)

    # Get exact answer string, skip over letter in parens if present
    # match = re.findall(r"The answer is\s+(\([a-zA-Z]\)\s+)?(.*)$", output)
    match = re.findall(r"Final Answer:\s+(\([a-zA-Z]\)\s+)?(.*)$", output)
    if not match:
        return output

    res = match[-1][-1].strip()
    if not res.endswith(".M."):
        res = res.strip(".")
    # Map single letter prediction to option
    if options and re.match(r"^[a-zA-Z]$", res):
        return next((
            option for option_idx, option in enumerate(options)
            if res == OPTION_INDS[option_idx]
        ), output)
    return res

def tabmwp_check_correct(src_meta_data: dict, pred_text: str):
    pred = extract_prediction(pred_text, src_meta_data["choices"])
    pred_norm = normalize_answer(pred, src_meta_data["unit"])
    label_norm = normalize_answer(src_meta_data["answer"], src_meta_data["unit"])
    return pred_norm.lower() == label_norm.lower()
