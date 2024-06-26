from typing import List
import random
from transformers import BertForNextSentencePrediction, AutoTokenizer, Trainer, TrainingArguments, BatchEncoding
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from reticl.data_loading.data_types import DatasetConfig
from reticl.utils import device, TrainOptions

class BERTDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig, split: str, options: TrainOptions):
        super().__init__()
        self.problems = []
        self.solutions = []
        for sample in tqdm(dataset_config["get_data"](split, options)[0]):
            sample = dataset_config["process_sample"](sample)
            self.problems.append(sample["encoder_context"])
            self.solutions.append(sample["encoder_label"])

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, index):
        if random.random() < 0.5:
            label_idx = index
        else:
            label_idx = random.randint(0, len(self.problems) - 1)
        return {
            "problem": self.problems[index],
            "solution": self.solutions[label_idx],
            "label": 0 if index == label_idx else 1
        }

class BERTCollator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def __call__(self, batch: List[dict]):
        return {
            **self.tokenizer(
                [item["problem"] for item in batch], [item["solution"] for item in batch],
                return_tensors="pt", padding=True, truncation=True, max_length=512
            ),
            "labels": torch.LongTensor([item["label"] for item in batch])
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    return {
        "accuracy": (predictions == labels).sum().item() / len(predictions)
    }

def finetune_bert(dataset_config: DatasetConfig, options_dict: dict):
    options = TrainOptions(options_dict)
    model = BertForNextSentencePrediction.from_pretrained("bert-base-cased").to(device)

    train_dataset = BERTDataset(dataset_config, "train", options)
    val_dataset = BERTDataset(dataset_config, "dev", options)

    training_args = TrainingArguments(
        output_dir=options.model_name,
        num_train_epochs=options.epochs,
        learning_rate=options.lr,
        weight_decay=options.wd,
        per_device_train_batch_size=options.batch_size,
        gradient_accumulation_steps=options.grad_accum_steps,
        per_device_eval_batch_size=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=BERTCollator()
    )
    trainer.train()
    trainer.save_model()
