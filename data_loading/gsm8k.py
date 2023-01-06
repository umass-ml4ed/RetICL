from typing import Optional
import json
import random
import re

from data_loading.data_loading import DatasetBase, DataSample
from models.retriever import Retriever
from utils import TrainOptions

def get_data(split: str):
    with open(f"grade-school-math/grade_school_math/data/{split}.jsonl") as data_file:
        samples = [json.loads(line) for line in data_file]
        random.Random(221).shuffle(samples)
        return samples

class GSM8KDataset(DatasetBase):
    def __init__(self, split: str, retriever: Optional[Retriever], options: TrainOptions):
        all_train_data = get_data("train")
        train_data, val_data = all_train_data[:-1000], all_train_data[-1000:]
        if split == "train":
            # Get training samples and corpus from train set
            data = train_data[:options.train_size]
            if options.corpus_size:
                corpus = train_data[options.train_size : options.train_size + options.corpus_size]
            else:
                corpus = train_data[options.train_size:]
        else:
            # Get evaluation samples from split and corpus from train set
            if split == "dev":
                data = val_data[:100]
            elif split == "dev1k":
                data = val_data
            else:
                data = get_data(split)
            # TODO: for not full corpus, should we use the same corpus for testing as we did for training?
            # corpus = train_data
            if options.corpus_size:
                corpus = random.Random(221).sample(train_data, options.corpus_size)
            else:
                corpus = train_data
        super().__init__(data, corpus, retriever, options)

    def process_sample(self, sample) -> DataSample:
        question = sample["question"].replace("\n", "\\n")
        answer = remove_calc_annotations(sample["answer"]).replace("\n", "\\n")
        return {
            "context": f"Question: {question}\nAnswer: ",
            "label": answer,
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

def extract_answer(solution: str):
    match = re.findall(r"#### (.*)$", solution)
    if not match:
        return ""
    return match[-1].replace(",", "").strip()

def gsm8k_check_correct(src_meta_data: dict, pred_text: str):
    return extract_answer(src_meta_data["answer"]) == extract_answer(pred_text)
