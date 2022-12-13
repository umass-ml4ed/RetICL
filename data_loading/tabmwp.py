from typing import Optional
import json
import random

from data_loading.data_loading import DatasetBase, DataSample
from models.retriever import Retriever
from promptPG.base_prompt import get_table_text, get_question_text, get_answer, get_solution_text
from constants import OPTION_INDS
from utils import TrainOptions

def get_data(split: str, samples_to_keep: int = 0):
    with open(f"tabmwp/problems_{split}.json", encoding="utf-8") as data_file:
        samples = list(json.load(data_file).values())
        if samples_to_keep:
            return random.Random(221).sample(samples, samples_to_keep)
        random.Random(221).shuffle(samples)
        return samples

class TabMWPDataset(DatasetBase):
    def __init__(self, split: str, retriever: Optional[Retriever], options: TrainOptions):
        if split == "train":
            # Get training samples and corpus from train set
            samples_to_keep = (options.train_size + options.corpus_size) if options.corpus_size else 0
            all_data = get_data("train", samples_to_keep)
            data, corpus = all_data[:options.train_size], all_data[options.train_size:]
        else:
            # Get evaluation samples from split and corpus from train set
            if split == "dev":
                data = get_data("dev", 100)
            else:
                data = get_data(split)
            corpus = get_data("train", options.corpus_size)
        super().__init__(data, corpus, retriever, options)

    def process_sample(self, sample) -> DataSample:
        # Prompt structure follows from PromptPG paper
        table = get_table_text(sample)
        question = get_question_text(sample, OPTION_INDS)
        answer = get_answer(sample)
        solution = get_solution_text(sample)
        return {
            "context": f"Table: {table}\nQuestion: {question}\nAnswer: ",
            "label": f"{solution} The answer is {answer}.",
            "meta_data": sample,
        }
