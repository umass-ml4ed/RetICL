import json
import os
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.gpt3 import gpt3_completion
from utils import device

def get_saved_cache(cache_filename: str):
    if os.path.exists(cache_filename):
        with open(cache_filename, encoding="utf-8") as cache_file:
            return json.load(cache_file)
    return {}

OPENAI_BATCH_SIZE = 10

class Generator:
    _cache: Dict[str, str] = {}
    _model_name = ""
    _gpt3_model_name = ""
    _model = None
    _tokenizer = None
    _newline_token = None
    _cache_filename = ""

    @classmethod
    def load(cls, args: dict):
        cls._model_name = args["generator_model"]
        if cls._model_name == "gpt3":
            cls._gpt3_model_name = args["gpt3_model"]
            cls._cache_filename = f"generator_cache_{cls._gpt3_model_name}.json"
        else:
            cls._cache_filename = f"generator_cache_{cls._model_name.replace('/', '-')}.json"
        cls._cache = get_saved_cache(cls._cache_filename)
        if cls._model_name != "gpt3":
            cls._model = AutoModelForCausalLM.from_pretrained(cls._model_name).to(device)
            cls._model.eval()
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
            cls._tokenizer.pad_token = cls._tokenizer.eos_token
            cls._newline_token = cls._tokenizer("\n").input_ids[0]

    @classmethod
    def save_cached(cls):
        # Get updates from other processes and then save whole thing
        temp_cache = get_saved_cache(cls._cache_filename)
        cls._cache.update(temp_cache)
        with open(cls._cache_filename, "w", encoding="utf-8") as cache_file:
            json.dump(cls._cache, cache_file, indent=2, ensure_ascii=False)

    @classmethod
    def get_ppl(cls, prompts: List[str], labels: List[str], **kwargs):
        if cls._model_name == "gpt3":
            raise ValueError("PPL not implemented for GPT-3")
        ppls = []
        for prompt, label in zip(prompts, labels):
            # Get tokens for model input
            prompt_inputs = cls._tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            label_inputs = cls._tokenizer(label, return_tensors="pt").input_ids.to(device)
            full_input = torch.cat([prompt_inputs, label_inputs], dim=1)
            # Get label tokens - don't calculate loss over prompt region
            label_tokens = torch.cat([torch.full_like(prompt_inputs, -100, device=device), label_inputs], dim=1)
            # In case we made the prompt too long, remove extra tokens from the beginning (retain whole label)
            if full_input.shape[1] >= cls._model.config.n_positions:
                overflow = full_input.shape[1] - cls._model.config.n_positions
                full_input = full_input[:, overflow:]
                label_tokens = label_tokens[:, overflow:]
            # Get loss from model
            with torch.no_grad():
                outputs = cls._model(
                    input_ids=full_input,
                    labels=label_tokens
                )
            loss = outputs.loss
            # Convert to perplexity
            ppls.append(torch.exp(loss))
        return torch.stack(ppls)

    @classmethod
    def generate(cls, prompts: List[str], **kwargs):
        if cls._model_name == "gpt3":
            uncached_prompts = [prompt for prompt in prompts if prompt not in cls._cache]
            if uncached_prompts:
                results = []
                for start_idx in range(0, len(uncached_prompts), OPENAI_BATCH_SIZE):
                    batch = uncached_prompts[start_idx : start_idx + OPENAI_BATCH_SIZE]
                    results += gpt3_completion(batch, cls._gpt3_model_name)
                for prompt, result in zip(uncached_prompts, results):
                    cls._cache[prompt] = result
        else:
            with torch.no_grad():
                for prompt in prompts:
                    if prompt in cls._cache:
                        continue
                    input_ids = cls._tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    # If we made the prompt too long, remove from beginning to make room for answer at the end
                    if input_ids.shape[1] >= cls._model.config.n_positions:
                        input_ids = input_ids[:, (input_ids.shape[1] - cls._model.config.n_positions) + 200:]
                    output = cls._model.generate(
                        input_ids,
                        pad_token_id=cls._tokenizer.eos_token_id,
                        eos_token_id=cls._newline_token,
                        # max_length=cls._model.config.n_positions,
                        max_new_tokens=200,
                        do_sample=False,
                    )
                    output = output[:, input_ids.shape[-1]:] # Just return generated portion
                    pred = cls._tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                    cls._cache[prompt] = pred
        return [cls._cache[prompt] for prompt in prompts]

class GeneratorCM:
    def __init__(self, args: dict):
        self.args = args

    def __enter__(self):
        Generator.load(self.args)

    def __exit__(self, exc_type, exc_value, traceback):
        Generator.save_cached()
