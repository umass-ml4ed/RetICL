from typing import List
import json
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from tqdm import tqdm
import wandb

from reticl.data_loading.pretrain_dataset import PreloadedSample, PretrainDataset
from reticl.data_loading.reticl_dataset import RetICLDataset, Collator, CollatedBatch
from reticl.data_loading.data_types import DatasetConfig
from reticl.models.retriever import retriever_model, Retriever
from reticl.training.train_reticl import get_rewards, get_predictions, get_optim
from reticl.utils import TrainOptions, device
from reticl.constants import SamplingMethod

# TODO for pre-training and baseline
# (maybe) Enable fine-tuning encoder on large corpus by getting required encodings in collator
# (maybe) Implement random sampling and ranking at test time for baseline

def get_suffix(options: TrainOptions):
    model_name = options.gpt3_model if options.generator_model == "gpt3" else options.generator_model
    return f"{options.dataset}_{model_name}_{options.num_examples}"

def collect_samples(split: str, dataset_config: DatasetConfig, options_dict: dict):
    options = TrainOptions(options_dict)
    assert(options.pt_sample_freq)
    options.sm = SamplingMethod.RANDOM.value
    dataset = RetICLDataset(dataset_config, split, None, options)
    data_loader = DataLoader(
        dataset,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=False,
        drop_last=False
    )
    results: List[PreloadedSample] = []
    for it in range(options.pt_sample_freq):
        print("Iteration:", it + 1)
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            predictions, _ = get_predictions(batch, dataset_config)
            for i in range(len(predictions)):
                results.append({
                    "prompt": batch["prompts"][i],
                    "label": batch["labels"][i],
                    "output": predictions[i],
                    "input_metadata": batch["meta_data"][i],
                    "example_metadatas": [
                        dataset.corpus[example_idx]["meta_data"]
                        for example_idx in batch["policy_example_indices"][i]
                    ],
                    "input_index": batch_idx * options.batch_size + i,
                    "policy_example_indices": batch["policy_example_indices"][i].tolist()
                })
    with open(f"reticl_pretrain_data_{split}_{get_suffix(options)}.json", "w") as out_f:
        json.dump(results, out_f, indent=2, ensure_ascii=False)

def pretrain_reticl(dataset_config: DatasetConfig, options_dict: dict):
    options = TrainOptions(options_dict)
    assert(options.pt_model_name)
    if options.wandb:
        run = wandb.init(project="reticl", config=options.as_dict())
    else:
        run = None

    retriever = retriever_model(options)
    best_model = retriever_model(options)
    optim, _, _ = get_optim([retriever], options)

    # Create train/val datasets/loaders
    with open(f"reticl_pretrain_data_train_{get_suffix(options)}.json") as in_f:
        train_samples = json.load(in_f)
    with open(f"reticl_pretrain_data_dev_{get_suffix(options)}.json") as in_f:
        val_samples = json.load(in_f)
    dataset = PretrainDataset(train_samples, dataset_config, "train", options)
    val_set = PretrainDataset(val_samples, dataset_config, "dev", options)
    data_loader = DataLoader(
        dataset,
        # Actual collate done outside loader so it's easier to collect samples for adding to replay buffer
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_set,
        collate_fn=Collator(),
        batch_size=options.batch_size,
        shuffle=False,
        drop_last=False
    )

    print("Training...")
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_val_loss = None
    for epoch in range(options.epochs):
        retriever.train()
        total_train_loss = 0
        for batch in tqdm(data_loader):
            _, correct = get_predictions(batch, dataset_config)
            returns = correct.repeat_interleave(options.num_examples).to(device)
            value_estimates = retriever.get_vfe(batch["current_sample_encodings"], batch["example_encodings"])
            value_estimates = value_estimates[:, 1:] # Don't train on initial state since doesn't impact example selection
            loss = loss_fn(value_estimates.contiguous().view(-1), returns)
            total_train_loss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()

        total_val_loss = 0
        total_val_correct = 0
        true_correct = 0
        retriever.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                _, correct = get_predictions(batch, dataset_config)
                true_correct += correct.sum().item()
                returns = correct.repeat_interleave(options.num_examples).to(device)
                value_estimates = retriever.get_vfe(batch["current_sample_encodings"], batch["example_encodings"])
                value_estimates = value_estimates[:, 1:]
                loss = loss_fn(value_estimates.contiguous().view(-1), returns)
                total_val_loss += loss.item()
                hard_predictions = value_estimates[:, -1].detach().cpu()
                hard_predictions[hard_predictions >= 0] = 1
                hard_predictions[hard_predictions < 0] = 0
                total_val_correct += (hard_predictions == correct).sum().item()

        avg_train_loss = total_train_loss / len(dataset)
        avg_val_loss = total_val_loss / len(val_set)
        avg_val_acc = total_val_correct / len(val_set)
        avg_true_correct = true_correct / len(val_set)
        if run:
            run.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": avg_val_acc
            })
        print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, True Correct: {avg_true_correct:.4f}")

        if not best_val_loss or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Best! Saving model...")
            best_model.load_state_dict(retriever.state_dict())
            torch.save(best_model.state_dict(), f"{options.pt_model_name}.pt")
