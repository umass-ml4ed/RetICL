from typing import List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
import pandas
import numpy as np

from models.retriever import Retriever, retriever_model
from models.generator import Generator
from data_loading.reticl import RetICLDataset, Collator
from data_loading.tabmwp import tabmwp_get_data, tabmwp_process_sample, tabmwp_check_correct
from data_loading.gsm8k import gsm8k_get_data, gsm8k_process_sample, gsm8k_check_correct
from constants import Datasets
from utils import TrainOptions

def check_correct(src_meta_data: dict, pred_text: str, options: TrainOptions):
    if options.dataset == Datasets.TABMWP.value:
        return tabmwp_check_correct(src_meta_data, pred_text)
    elif options.dataset == Datasets.GSM8K.value:
        return gsm8k_check_correct(src_meta_data, pred_text)
    raise Exception(f"Dataset {options.dataset} not supported!")

def evaluate(run, retriever: Optional[Retriever], split: str, options: TrainOptions):
    if not run and options.wandb:
        run = wandb.init(project="reticl", config=options.as_dict())
    if retriever:
        retriever.eval()
    if options.dataset == Datasets.TABMWP.value:
        dataset = RetICLDataset(tabmwp_get_data, tabmwp_process_sample, split, retriever, options)
    elif options.dataset == Datasets.GSM8K.value:
        dataset = RetICLDataset(gsm8k_get_data, gsm8k_process_sample, split, retriever, options)
    else:
        raise Exception(f"Dataset {options.dataset} not supported!")
    dataset.set_greedy(True) # Use greedy sampling for policy-based example retrieval
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=False
    )
    prompts: List[str] = []
    labels: List[str] = []
    meta_datas: List[dict] = []
    preds: List[str] = []
    example_set = set()
    # Sample batch from dataset - example retrieval is done by __getitem__ in RetICLDataset
    for batch in tqdm(data_loader):
        for example_idx in batch["policy_example_indices"].view(-1):
            example_set.add(example_idx.item())

        prompts += batch["prompts"]
        labels += batch["labels"]
        meta_datas += batch["meta_data"]
        preds += Generator.generate(**batch)

    correct = np.array([check_correct(meta_data, pred, options) for meta_data, pred in zip(meta_datas, preds)])
    acc = correct.mean() * 100
    if options.dataset == Datasets.TABMWP.value:
        mc_acc = correct[[meta_data["ques_type"] == "multi_choice" for meta_data in meta_datas]].mean() * 100
        free_acc = correct[[meta_data["ques_type"] == "free_text" for meta_data in meta_datas]].mean() * 100
        print(f"Accuracy: {acc:.2f}, MC: {mc_acc:.2f}, Free: {free_acc:.2f}, Examples: {len(example_set)}")
    else:
        print(f"Accuracy: {acc:.2f}, Examples: {len(example_set)}")
    if run:
        run.config.eval_set = split
        run.summary["accuracy"] = acc
        if options.dataset == Datasets.TABMWP.value:
            run.summary["mc_accuracy"] = mc_acc
            run.summary["free_accuracy"] = free_acc

    model_name = options.model_name if options.model_name else\
        f"{options.sm}_{options.generator_model}" + (f"_{options.gpt3_model}" if options.generator_model == "gpt3" else "")
    out_filename = f"results_{options.dataset}_{split}_{model_name}.csv"
    df = pandas.DataFrame({
        "prompt": prompts,
        "label": labels,
        "pred": preds,
        "correct": correct,
    })
    if options.dataset == Datasets.TABMWP.value:
        df["type"] = [meta_data["ques_type"] for meta_data in meta_datas]
    df.to_csv(out_filename)

def evaluate_reticl(options_dict: dict, split: str):
    options = TrainOptions(options_dict)
    if options.model_name:
        retriever = retriever_model(options)
        retriever.load_state_dict(torch.load(f"{options.model_name}.pt"))
    else:
        retriever = None
    evaluate(None, retriever, split, options)

def answer_missing(df: pandas.DataFrame, dataset: str):
    inidcator = "The answer is " if dataset == Datasets.TABMWP.value else "#### "
    return (len(df) - df["pred"].str.contains(inidcator).sum()) / len(df)

def error_analysis(group_str: str, result_file_1: str, result_file_2: str, arg_dict: dict):
    num_examples = 10
    df_1 = pandas.read_csv(result_file_1)
    df_2 = pandas.read_csv(result_file_2)
    # print("Pct left missing answer:", answer_missing(df_1, arg_dict["dataset"]))
    # print("Pct right missing answer:", answer_missing(df_2, arg_dict["dataset"]))
    if group_str == "left":
        group = df_1["correct"] & ~df_2["correct"]
    elif group_str == "right":
        group = ~df_1["correct"] & df_2["correct"]
    elif group_str == "both":
        group = df_1["correct"] & df_2["correct"]
    elif group_str == "neither":
        group = ~df_1["correct"] & ~df_2["correct"]
    for ((_, row_1), (_, row_2)) in zip(df_1[group][:num_examples].iterrows(), df_2[group][:num_examples].iterrows()):
        print("Method 1:\n", row_1["prompt"] + row_1["pred"], "\n")
        print("Method 2:\n", row_2["prompt"] + row_2["pred"], "\n")
        print()
