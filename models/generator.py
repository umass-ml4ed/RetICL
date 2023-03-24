import json
import os
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, PreTrainedTokenizerBase
import deepspeed

from models.gpt3 import gpt3_completion_parallel, gpt3_completion_with_batching
from utils import device, TrainOptions

USE_DS = True

def get_saved_cache(cache_filename: str):
    if os.path.exists(cache_filename):
        with open(cache_filename, encoding="utf-8") as cache_file:
            return json.load(cache_file)
    return {}

class NLStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, input_len: int):
        self.nl_token = tokenizer("\n").input_ids[0]
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor, **_kwargs) -> bool:
        # Check if two conginuous newlines exist in all sequences
        return all([
            any([
                seq[idx] == self.nl_token and seq[idx + 1] == self.nl_token
                for idx in range(self.input_len, input_ids.shape[1] - 1)
            ])
            for seq in input_ids
        ])

class Generator:
    _cache: Dict[str, str] = {}
    _model_name = ""
    _gpt3_model_name = ""
    _model = None
    _tokenizer = None
    _cache_filename = ""

    @classmethod
    def load(cls, args: dict):
        cls.options = TrainOptions(args)
        cls._model_name = cls.options.generator_model
        if cls._model_name == "gpt3":
            cls._gpt3_model_name = cls.options.gpt3_model
            model_name = cls._gpt3_model_name
        else:
            model_name = cls._model_name.replace("/", "-")
        cls._cache_filename = f"generator_cache_{cls.options.dataset}_{model_name}_ex{cls.options.num_examples}_mgt{cls.options.max_gen_tokens}.json"
        cls._cache = get_saved_cache(cls._cache_filename)
        if cls._model_name != "gpt3":
            print("Loading generator model...")
            if USE_DS:
                if cls._model_name == "EleutherAI/gpt-j-6B":
                    model_name = "philschmid/gpt-j-6B-fp16-sharded"
                else:
                    model_name = cls._model_name
                cls._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
                cls._model = deepspeed.init_inference(
                    model=cls._model,
                    mp_size=1,
                    dtype=torch.float16,
                    replace_with_kernel_inject=True,
                )
            else:
                cls._model = AutoModelForCausalLM.from_pretrained(cls._model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
                if "gpt-neox" in cls._model_name:
                    cls._model = cls._model.half()
                cls._model = cls._model.to(device)
            cls._model.eval()
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
            cls._tokenizer.padding_side = "left"
            cls._tokenizer.pad_token = cls._tokenizer.eos_token

    @classmethod
    def save_cached(cls):
        # Get updates from other processes and then save whole thing
        temp_cache = get_saved_cache(cls._cache_filename)
        cls._cache.update(temp_cache)
        print(f"Saving cache ({len(cls._cache)} entries)...")
        with open(cls._cache_filename, "w", encoding="utf-8") as cache_file:
            json.dump(cls._cache, cache_file, indent=2, ensure_ascii=False)

    @classmethod
    def get_nll(cls, prompts: List[str], labels: List[str], **kwargs):
        if cls._model_name == "gpt3":
            # TODO: delete this...
            import pdb; pdb.set_trace()
            full_text = [prompt + label for prompt, label in zip(prompts, labels)]
            results = gpt3_completion_with_batching(full_text, cls._gpt3_model_name, max_tokens=0, logprobs=1, echo=True)
            nlls = []
            for result_idx, choice in enumerate(results):
                # prompt_len = len(prompts[result_idx])
                # running_len = 0
                # for token_idx, token in enumerate(choice["logprobs"]["tokens"]):
                #     running_len += len(token)
                #     if running_len > prompt_len:
                #         nlls.append(-torch.tensor(choice["logprobs"]["token_logprobs"][token_idx:]).mean())
                #         break
                for token_idx in range(len(choice["logprobs"]["tokens"]), -1, -1):
                    if choice["logprobs"]["tokens"][token_idx - 1] == ":" and\
                        choice["logprobs"]["tokens"][token_idx - 2] == "Solution":
                        nlls.append(-torch.tensor(choice["logprobs"]["token_logprobs"][token_idx:]).mean())
                        break
            return torch.stack(nlls)
        nlls = []
        for prompt, label in zip(prompts, labels):
            # Get tokens for model input
            # TODO: left padded tokenizer might not work here
            prompt_inputs = cls._tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            label_inputs = cls._tokenizer(label, return_tensors="pt").input_ids.to(device)
            full_input = torch.cat([prompt_inputs, label_inputs], dim=1)
            # Get label tokens - don't calculate loss over prompt region
            label_tokens = torch.cat([torch.full_like(prompt_inputs, -100, device=device), label_inputs], dim=1)
            # In case we made the prompt too long, remove extra tokens from the beginning (retain whole label)
            if "gpt-j" in cls._model_name:
                max_tokens = cls._model.config.n_positions
            else:
                max_tokens = cls._model.config.max_position_embeddings
            if full_input.shape[1] >= max_tokens:
                overflow = full_input.shape[1] - max_tokens
                full_input = full_input[:, overflow:]
                label_tokens = label_tokens[:, overflow:]
            # Get loss from model
            with torch.no_grad():
                outputs = cls._model(
                    input_ids=full_input,
                    labels=label_tokens
                )
            nlls.append(outputs.loss)
        return torch.stack(nlls).float()

    @classmethod
    def generate(cls, prompts: List[str], **kwargs):
        if cls._model_name == "gpt3":
            uncached_prompts = [prompt for prompt in prompts if prompt not in cls._cache]
            if uncached_prompts:
                results = gpt3_completion_with_batching(uncached_prompts, cls._gpt3_model_name, cls.options.max_gen_tokens)
                assert len(uncached_prompts) == len(results)
                for prompt, result in zip(uncached_prompts, results):
                    if "gpt-3.5-turbo" in cls._gpt3_model_name:
                        cls._cache[prompt] = result["message"]["content"]
                    else:
                        cls._cache[prompt] = result["text"]
        else:
            with torch.no_grad():
                if USE_DS:
                    uncached_prompts = [prompt for prompt in prompts if prompt not in cls._cache]
                    if uncached_prompts:
                        inputs = cls._tokenizer(uncached_prompts, return_tensors="pt", padding=True).to(device)
                        overflow = inputs.input_ids.shape[1] + cls.options.max_gen_tokens - 1024
                        if overflow > 0:
                            # Remove tokens from beginning if prompt is too long
                            # inputs.input_ids = inputs.input_ids[:, overflow:]
                            # inputs.attention_mask = inputs.attention_mask[:, overflow:]
                            print("Too long!")
                        outputs = cls._model.generate(
                            **inputs,
                            pad_token_id=cls._tokenizer.eos_token_id,
                            stopping_criteria=[NLStoppingCriteria(cls._tokenizer, inputs.input_ids.shape[1])],
                            max_new_tokens=cls.options.max_gen_tokens,
                            do_sample=False,
                            # top_k=4,
                            # penalty_alpha=0.6,
                        )
                        outputs = outputs[:, inputs.input_ids.shape[1]:]
                        preds = cls._tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        for prompt, pred in zip(uncached_prompts, preds):
                            first_nl = pred.find("\n\n")
                            if first_nl != -1:
                                pred = pred[:first_nl]
                            cls._cache[prompt] = pred
                else:
                    for prompt in prompts:
                        if prompt in cls._cache:
                            continue
                        input_ids = cls._tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                        # If we made the prompt too long, remove from beginning to make room for answer at the end
                        if "gpt-j" in cls._model_name:
                            max_tokens = cls._model.config.n_positions
                        else:
                            max_tokens = cls._model.config.max_position_embeddings
                        if input_ids.shape[1] >= max_tokens:
                            input_ids = input_ids[:, (input_ids.shape[1] - max_tokens) + cls.options.max_gen_tokens:]
                        output = cls._model.generate(
                            input_ids,
                            pad_token_id=cls._tokenizer.eos_token_id,
                            stopping_criteria=[NLStoppingCriteria(cls._tokenizer, input_ids.shape[1])],
                            max_new_tokens=cls.options.max_gen_tokens,
                            do_sample=False,
                            num_beams=4,
                        )
                        output = output[:, input_ids.shape[-1]:] # Just return generated portion
                        pred = cls._tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                        first_nl = pred.find("\n\n")
                        if first_nl != -1:
                            pred = pred[:first_nl]
                        cls._cache[prompt] = pred
        return [cls._cache[prompt] for prompt in prompts]

class GeneratorCM:
    def __init__(self, args: dict):
        self.args = args

    def __enter__(self):
        Generator.load(self.args)

    def __exit__(self, exc_type, exc_value, traceback):
        Generator.save_cached()
