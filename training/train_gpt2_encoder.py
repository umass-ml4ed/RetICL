from typing import List
from transformers import GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments, BatchEncoding
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from data_loading.data_types import GetDataFunction, ProcessDataFunction
from utils import device, TrainOptions

class GPT2Dataset(Dataset):
    def __init__(self, get_data: GetDataFunction, process_sample: ProcessDataFunction, split: str, options: TrainOptions):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        self.data: List[BatchEncoding] = []
        for sample in tqdm(get_data(split, options)[0]):
            sample = process_sample(sample)
            self.data.append(tokenizer(sample["encoder_context"] + sample["encoder_label"], return_tensors="pt"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class GPT2Collator:
    def __call__(self, batch: List[BatchEncoding]):
        return {
            "input_ids": pad_sequence([item.input_ids[0] for item in batch], batch_first=True, padding_value=50256),
            "labels": pad_sequence([item.input_ids[0] for item in batch], batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence([item.attention_mask[0] for item in batch], batch_first=True, padding_value=0)
        }

def finetune_gpt2(get_data: GetDataFunction, process_sample: ProcessDataFunction, options_dict: dict):
    options = TrainOptions(options_dict)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    train_dataset = GPT2Dataset(get_data, process_sample, "train", options)
    val_dataset = GPT2Dataset(get_data, process_sample, "dev1k", options)

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
        remove_unused_columns=False,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=GPT2Collator()
    )
    trainer.train()
    trainer.save_model()
