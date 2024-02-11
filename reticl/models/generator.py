import json
import os
import time
from typing import Dict, List, TypedDict, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, PreTrainedTokenizerBase
try:
    import deepspeed
except ModuleNotFoundError:
    pass

from reticl.models.gpt3 import gpt3_completion_parallel, gpt3_completion_with_batching
from reticl.utils import device, TrainOptions

class GeneratorResult(TypedDict):
    text: str
    nll: float

def get_saved_cache(cache_filename: str):
    if os.path.exists(cache_filename):
        with open(cache_filename, encoding="utf-8") as cache_file:
            return json.load(cache_file)
    return {}

def get_stopping_point(seq: torch.LongTensor, start: int, nl_token: int) -> Optional[int]:
    return next(
        (idx for idx in range(start, seq.shape[0] - 1)
            if seq[idx] == nl_token and seq[idx + 1] == nl_token),
        None
    )

class NLStoppingCriteria(StoppingCriteria):
    def __init__(self, nl_token: int, input_len: int):
        self.nl_token = nl_token
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor, **_kwargs) -> bool:
        # Check if two conginuous newlines exist in all sequences
        return all([
            get_stopping_point(seq, self.input_len, self.nl_token) is not None
            for seq in input_ids
        ])

class Generator:
    options = None
    _cache: Dict[str, GeneratorResult] = {}
    _model_name = ""
    _gpt3_model_name = ""
    _model = None
    _model_ds = None
    _use_ds = False
    _gen_batch_size = 0
    _tokenizer = None
    _max_tokens = 0
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
        if cls.options.gen_batch_size:
            cls._gen_batch_size = cls.options.gen_batch_size
        else:
            if cls._model_name == "gpt3":
                cls._gen_batch_size = 10
            elif "gpt-neox" in cls._model_name:
                cls._gen_batch_size = 2
            else:
                cls._gen_batch_size = 10
        if cls._model_name != "gpt3":
            print("Loading generator model...")
            start_time = time.time()
            # Load tokenizer and set newline token for double newline stopping criteria
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
            if "llama" in cls._model_name:
                cls._tokenizer.pad_token = cls._tokenizer.bos_token
                cls._nl_token = 13
            elif "gpt" in cls._model_name:
                cls._tokenizer.pad_token = cls._tokenizer.eos_token
                cls._nl_token = cls._tokenizer("\n").input_ids[0]
            else:
                raise NotImplementedError(f"Double newline tokenization not implemented for model {cls._model_name}")
            # Load model
            cls._model = AutoModelForCausalLM.from_pretrained(
                cls._model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if "gpt-j" in cls._model_name:
                cls._max_tokens = cls._model.config.n_positions
            else:
                cls._max_tokens = cls._model.config.max_position_embeddings
            cls._model.eval()
            cls._use_ds = "gpt-j" in cls._model_name or "gpt-neox" in cls._model_name
            if cls._use_ds:
                cls._model_ds = deepspeed.init_inference(
                    model=cls._model,
                    mp_size=1,
                    dtype=torch.float16,
                    replace_with_kernel_inject=False,
                    max_out_tokens=cls._max_tokens
                )            
            print(f"Generator model loaded ({time.time() - start_time:.2f}s)")

    @classmethod
    def save_cached(cls):
        # Get updates from other processes and then save whole thing
        temp_cache = get_saved_cache(cls._cache_filename)
        temp_cache.update(cls._cache)
        print(f"Saving cache ({len(temp_cache)} entries)...")
        with open(cls._cache_filename, "w", encoding="utf-8") as cache_file:
            json.dump(temp_cache, cache_file, indent=2, ensure_ascii=False)

    @classmethod
    def get_nll(cls, prompts: List[str], labels: List[str], **kwargs):
        if cls._model_name == "gpt3":
            full_text = [prompt + label for prompt, label in zip(prompts, labels)]
            results = gpt3_completion_with_batching(full_text, cls._gen_batch_size, cls._gpt3_model_name, max_tokens=0, logprobs=1, echo=True, verbose=cls.options.verbose)
            nlls = []
            for result_idx, choice in enumerate(results):
                # Find index in tokens where label begins and average per-token nlls thereafter
                prompt_len = len(prompts[result_idx].encode())
                prompt = bytes()
                for token_idx, token in enumerate(choice["logprobs"]["tokens"]):
                    if token.startswith("bytes:"):
                        # Convert literal byte representation to actual bytes object
                        # https://stackoverflow.com/questions/41552839/how-can-i-convert-literal-escape-sequences-in-a-string-to-the-corresponding-byte
                        prompt += token[6:].encode().decode("unicode_escape").encode("latin-1")
                    else:
                        prompt += token.encode()
                    if len(prompt) > prompt_len:
                        nlls.append(-torch.tensor(choice["logprobs"]["token_logprobs"][token_idx:]).mean())
                        break

            return torch.stack(nlls)

        cls._tokenizer.padding_side = "right"
        nlls = []
        for start_idx in range(0, len(prompts), cls._gen_batch_size):
            # Construct tokenized inputs
            prompt_batch = prompts[start_idx : start_idx + cls._gen_batch_size]
            label_batch = labels[start_idx : start_idx + cls._gen_batch_size]
            full_inputs = cls._tokenizer(
                [prompt + label for prompt, label in zip(prompt_batch, label_batch)],
                return_tensors="pt", padding=True, truncation=True, max_length=cls._max_tokens
            ).to(device)
            # Get labels - don't compute loss over prompt or padding regions
            prompt_attn_mask = cls._tokenizer(prompt_batch, return_tensors="pt", padding=True).attention_mask.to(device)
            prompt_attn_mask = F.pad(prompt_attn_mask, (0, full_inputs.input_ids.shape[1] - prompt_attn_mask.shape[1]), value=0)
            label_tokens = torch.clone(full_inputs.input_ids)
            loss_mask = (prompt_attn_mask == 0) & (full_inputs.attention_mask == 1)
            label_tokens[~loss_mask] = -100
            # Get loss from model
            with torch.no_grad():
                outputs = cls._model(
                    **full_inputs,
                    labels=label_tokens
                )
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = label_tokens[..., 1:].contiguous()
                loss = torch.nn.CrossEntropyLoss(reduction="none")(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1)
                ).view(shift_labels.shape)
                loss = loss.sum(dim=1) / loss_mask.sum(dim=1)
                nlls.append(loss)
        return torch.cat(nlls).float()

    @classmethod
    def generate(cls, prompts: List[str], **kwargs):
        if cls._model_name == "gpt3":
            uncached_prompts = [prompt for prompt in prompts if prompt not in cls._cache or isinstance(cls._cache[prompt], str)]
            if uncached_prompts:
                use_chat = cls._gpt3_model_name in ("gpt-3.5-turbo", "gpt-4")
                if use_chat:
                    results = gpt3_completion_parallel(uncached_prompts, 1, cls._gpt3_model_name, cls.options.max_gen_tokens, verbose=cls.options.verbose)
                else:
                    results = gpt3_completion_with_batching(uncached_prompts, cls._gen_batch_size, cls._gpt3_model_name, cls.options.max_gen_tokens, logprobs=1, verbose=cls.options.verbose)
                assert len(uncached_prompts) == len(results)
                for prompt, result in zip(uncached_prompts, results):
                    if use_chat:
                        cls._cache[prompt] = {
                            "text": result["message"]["content"],
                            "nll": None
                        }
                    else:
                        cls._cache[prompt] = {
                            "text": result["text"],
                            "nll": -torch.tensor(result["logprobs"]["token_logprobs"]).mean().item()
                        }
        else:
            torch.use_deterministic_algorithms(False) # Don't use deterministic algorithms to speed up generation
            with torch.no_grad():
                cls._tokenizer.padding_side = "left"
                uncached_prompts = [prompt for prompt in prompts if prompt not in cls._cache or isinstance(cls._cache[prompt], str)]
                if uncached_prompts:
                    for start_idx in range(0, len(uncached_prompts), cls._gen_batch_size):
                        batch = uncached_prompts[start_idx : start_idx + cls._gen_batch_size]
                        inputs = cls._tokenizer(
                            batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=cls._max_tokens - cls.options.max_gen_tokens).to(device)
                        model = cls._model_ds if cls._use_ds else cls._model
                        outputs = model.generate(
                            **inputs,
                            pad_token_id=cls._tokenizer.eos_token_id,
                            stopping_criteria=[NLStoppingCriteria(cls._nl_token, inputs.input_ids.shape[1])],
                            max_new_tokens=cls.options.max_gen_tokens,
                            do_sample=False,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        sequences = outputs.sequences[:, inputs.input_ids.shape[1]:] # Just return generated portion
                        scores = torch.stack(outputs.scores, dim=1) # Scores are only for generated portion
                        for prompt, seq, score in zip(batch, sequences, scores):
                            # Trim sequence and logits after first double newline
                            sp = get_stopping_point(seq, 0, cls._nl_token)
                            if sp is not None:
                                seq = seq[:sp]
                                score = score[:sp]
                            # Get predicted text from tokens and nll from logits
                            pred = cls._tokenizer.decode(seq, skip_special_tokens=True)
                            logps = F.log_softmax(score, dim=-1)
                            nll = -logps[torch.arange(seq.shape[0]), seq].mean().item()
                            cls._cache[prompt] = {
                                "text": pred,
                                "nll": nll
                            }
            torch.use_deterministic_algorithms(cls.options.deterministic, warn_only=True) # Set determinism back
        return [cls._cache[prompt] for prompt in prompts]

class GeneratorCM:
    def __init__(self, args: dict):
        self.args = args

    def __enter__(self):
        Generator.load(self.args)

    def __exit__(self, exc_type, exc_value, traceback):
        Generator.save_cached()
