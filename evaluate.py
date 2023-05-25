from typing import List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb
import pandas
import numpy as np

from models.retriever import Retriever, retriever_model
from models.generator import Generator
from data_loading.data_types import GetDataFunction, ProcessDataFunction, CheckCorrectFunction
from data_loading.reticl_dataset import RetICLDataset, Collator
from constants import Datasets, SamplingMethod
from utils import TrainOptions, device

def exhaustive_eval(dataset: RetICLDataset, check_correct: CheckCorrectFunction, options: TrainOptions):
    # Get maximum possible performance
    prompts: List[str] = []
    labels: List[str] = []
    meta_datas: List[dict] = []
    preds: List[str] = []
    for sample in tqdm(dataset.data):
        labels.append(sample["lm_label"])
        meta_datas.append(sample["meta_data"])

        def get_all_prompts(cur_prompt, cur_len):
            # Recursively get all possible prompts from all example permutations
            if cur_len == options.num_examples:
                return [cur_prompt + sample["lm_context"]]
            results = []
            for example in dataset.corpus:
                results += get_all_prompts(
                    cur_prompt + example["lm_context"] + example["lm_label"] + "\n\n",
                    cur_len + 1
                )
            return results

        # Query LM with possible prompts until correctness achieved
        correct_found = False
        all_prompt_cands = get_all_prompts("", 0)
        for prompt_idx in range(0, len(all_prompt_cands), options.batch_size):
            prompt_cands = all_prompt_cands[prompt_idx : prompt_idx + options.batch_size]
            pred_cands = [pred["text"] for pred in Generator.generate(prompts=prompt_cands)]
            correct = [check_correct(meta_datas[-1], pred) for pred in pred_cands]
            if any(correct):
                correct_found = True
                first_correct = np.argmax(correct)
                prompts.append(prompt_cands[first_correct])
                preds.append(pred_cands[first_correct])
                break
        if not correct_found:
            prompts.append(sample["lm_context"])
            preds.append("")

    return prompts, labels, meta_datas, preds

def policy_eval(dataset: RetICLDataset, options: TrainOptions):
    prompts: List[str] = []
    labels: List[str] = []
    meta_datas: List[dict] = []
    preds: List[str] = []
    example_set = set()
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=False
    )
    # Sample batch from dataset - example retrieval is done by __getitem__ in RetICLDataset
    for batch in tqdm(data_loader):
        for example_idx in batch["policy_example_indices"].view(-1):
            example_set.add(example_idx.item())

        prompts += batch["prompts"]
        labels += batch["labels"]
        meta_datas += batch["meta_data"]
        for pred in Generator.generate(**batch):
            preds.append(pred["text"])

    return prompts, labels, meta_datas, preds, example_set

def evaluate_reticl(run, get_data: GetDataFunction, process_sample: ProcessDataFunction, check_correct: CheckCorrectFunction,
             retriever: Optional[Retriever], split: str, options: TrainOptions):
    with torch.no_grad():
        if not run and options.wandb:
            run = wandb.init(project="reticl", config=options.as_dict())
        if retriever:
            retriever.eval()
        dataset = RetICLDataset(get_data, process_sample, split, retriever, options)
        dataset.set_greedy(True) # Use greedy sampling for policy-based example retrieval

        # Collect predictions and labels over the dataset
        if options.sm == SamplingMethod.EXHAUSTIVE.value:
            prompts, labels, meta_datas, preds = exhaustive_eval(dataset, check_correct, options)
            example_set = set()
        else:
            prompts, labels, meta_datas, preds, example_set = policy_eval(dataset, options)

        correct = np.array([check_correct(meta_data, pred) for meta_data, pred in zip(meta_datas, preds)])
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
            run.summary["eval_examples"] = len(example_set)
            if options.dataset == Datasets.TABMWP.value:
                run.summary["mc_accuracy"] = mc_acc
                run.summary["free_accuracy"] = free_acc

        generator_model = options.generator_model.replace("/", "-")
        model_name = options.model_name if options.model_name else\
            f"{options.sm}_{generator_model}" + (f"_{options.gpt3_model}" if options.generator_model == "gpt3" else "")
        out_filename = f"results_{options.dataset}_{split}_{model_name}_tex{options.num_examples}_mgt{options.max_gen_tokens}"
        if options.val_corpus_size:
            out_filename += f"_vcs{options.val_corpus_size}"
        out_filename += ".csv"
        df = pandas.DataFrame({
            "prompt": prompts,
            "label": labels,
            "pred": preds,
            "correct": correct,
        })
        if options.dataset == Datasets.TABMWP.value:
            df["type"] = [meta_data["ques_type"] for meta_data in meta_datas]
        df.to_csv(out_filename)

def evaluate(get_data: GetDataFunction, process_sample: ProcessDataFunction, check_correct: CheckCorrectFunction,
             split: str, options_dict: dict):
    options = TrainOptions(options_dict)
    if options.model_name:
        retriever = retriever_model(options)
        retriever.load_state_dict(torch.load(f"{options.model_name}.pt", map_location=device))
    else:
        retriever = None
    evaluate_reticl(None, get_data, process_sample, check_correct, retriever, split, options)

def answer_missing(df: pandas.DataFrame, dataset: str):
    # indicator = "The answer is " if dataset == Datasets.TABMWP.value else "Final Answer: "
    indicator = "Final Answer: "
    return (len(df) - df["pred"].str.contains(indicator).sum()) / len(df)

def error_analysis(title: str, result_file_1: str, result_file_2: str, arg_dict: dict):
    num_examples = 10
    df_1 = pandas.read_csv(result_file_1)
    df_2 = pandas.read_csv(result_file_2)
    # print("Pct left missing answer:", answer_missing(df_1, arg_dict["dataset"]))
    # print("Pct right missing answer:", answer_missing(df_2, arg_dict["dataset"]))
    df = df_1.join(df_2, lsuffix="_l", rsuffix="_r")
    df = df[["prompt_l", "pred_l", "prompt_r", "pred_r", "label_l", "correct_l", "correct_r"]]
    left = df[df["correct_l"] > df["correct_r"]]
    right = df[df["correct_l"] < df["correct_r"]]
    both = df[(df["correct_l"] == 1) & (df["correct_r"] == 1)]
    neither = df[(df["correct_l"] == 0) & (df["correct_r"] == 0)]
    for sub_title, sub_df in [("left", left), ("right", right), ("both", both), ("neither", neither)]:
        print(sub_title, len(sub_df))
        sub_df.to_csv(f"{title}_{sub_title}.csv")
