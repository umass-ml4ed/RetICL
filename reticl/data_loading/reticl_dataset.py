from typing import TypedDict, List, Optional
import random
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Model, BertModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

from reticl.data_loading.data_types import DataSample, DatasetConfig
from reticl.models.retriever import Retriever
from reticl.models.encoder import SBERTEncoder
from reticl.models.generator import GeneratorResult
from reticl.constants import SamplingMethod, EncoderModelType
from reticl.utils import device, TrainOptions

class ICLSample(TypedDict):
    # For evaluation
    prompt: str
    label: str
    output: Optional[GeneratorResult]
    meta_data: dict
    # For retriever
    current_sample_encoding: torch.Tensor
    example_encodings: torch.Tensor
    policy_example_indices: torch.Tensor
    # For PG version of model
    all_example_encodings: torch.Tensor

class CollatedBatch(TypedDict):
    # For evaluation
    prompts: List[str]
    labels: List[str]
    outputs: Optional[List[GeneratorResult]]
    meta_data: List[dict]
    # For retriever
    current_sample_encodings: torch.Tensor
    example_encodings: torch.Tensor
    policy_example_indices: torch.Tensor
    seq_len: torch.Tensor
    # For PG version of model
    all_example_encodings: torch.Tensor

class RetICLDataset(TorchDataset):
    def __init__(self, dataset_config: DatasetConfig,
                 split: str, retriever: Optional[Retriever], options: TrainOptions, compute_intial_encodings: bool = True,
                 cached_encoding_matrix: Optional[torch.Tensor] = None):
        super().__init__()

        self.options = options
        self.retriever = retriever
        self.epsilon = 0.0
        self.greedy = False

        # Process data samples
        samples, corpus = dataset_config["get_data"](split, options)
        self.data = [dataset_config["process_sample"](sample) for sample in samples]
        if corpus:
            self.corpus = [dataset_config["process_sample"](sample) for sample in corpus]
        else:
            self.corpus = self.data
        self.prompt_prefix = dataset_config.get("prompt_prefix")

        # Compute encodings
        self.trainable_encoder = False
        if options.sm not in (SamplingMethod.RANDOM.value, SamplingMethod.COMPLEX.value):
            if retriever is not None and retriever.encoder is not None:
                # We have a trainable encoder, so encodings are computed on the fly
                self.trainable_encoder = True
                self.encoder = retriever.encoder
                if compute_intial_encodings:
                    self.compute_corpus_encodings(cached_encoding_matrix=cached_encoding_matrix)
            else:
                # We have a static encoder, so compute encodings now
                if self.options.encoder_model_type == EncoderModelType.BERT.value:
                    self.encoder = BertModel.from_pretrained(self.options.encoder_model or "bert-base-cased").to(device)
                    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
                elif self.options.encoder_model_type == EncoderModelType.GPT2.value:
                    self.encoder = GPT2Model.from_pretrained(self.options.encoder_model or "gpt2").to(device)
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.encoder = SentenceTransformer(self.options.encoder_model or "all-distilroberta-v1")
                self.compute_encodings(cached_encoding_matrix=cached_encoding_matrix)

        self.eos_idx = len(self.corpus)

        if options.sm == SamplingMethod.COMPLEX.value:
            complexity_metric = dataset_config.get("complexity_metric", lambda sample: sample["lm_label"].count("\n"))
            corpus_complexity = [-complexity_metric(sample) for sample in self.corpus]
            self.complex_example_idxs = np.flip(np.argsort(corpus_complexity)[:options.num_examples]).copy()

    def batch_encode(self, samples: List[DataSample], inc_label: bool, show_progress: bool = True):
        batch_size = 10
        it = range(0, len(samples), batch_size)
        if show_progress:
            it = tqdm(it)
        for batch_start_idx in it:
            batch = samples[batch_start_idx : batch_start_idx + batch_size]
            if self.options.encoder_model_type == EncoderModelType.SBERT.value:
                seq_strings = [
                    (sample["encoder_context"] + sample["encoder_label"]) if inc_label else sample["encoder_context"]
                    for sample in batch
                ]
                if self.trainable_encoder:
                    assert isinstance(self.encoder, SBERTEncoder)
                    outputs = self.encoder.encode(seq_strings, inc_label)
                else:
                    outputs = self.encoder.encode(seq_strings, convert_to_tensor=True, normalize_embeddings=True)
            elif self.options.encoder_model_type == EncoderModelType.BERT.value:
                inputs = self.tokenizer(
                    [sample["encoder_context"] for sample in batch],
                    [sample["encoder_label"] for sample in batch] if inc_label else None,
                    return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(device)
                outputs = self.encoder(**inputs).pooler_output
            elif self.options.encoder_model_type == EncoderModelType.GPT2.value:
                seq_strings = [
                    (sample["encoder_context"] + sample["encoder_label"]) if inc_label else sample["encoder_context"]
                    for sample in batch
                ]
                inputs = self.tokenizer(seq_strings, return_tensors="pt", padding=True).to(device)
                outputs = self.encoder(**inputs).last_hidden_state[
                    torch.arange(inputs.input_ids.shape[0]),
                    torch.sum(inputs.attention_mask, dim=-1) - 1
                ]
            for sample, encoding in zip(batch, outputs):
                if inc_label:
                    sample["full_encoding"] = encoding
                else:
                    sample["context_encoding"] = encoding

    def compute_corpus_encodings(self, show_progress: bool = True, cached_encoding_matrix: torch.Tensor = None):
        if cached_encoding_matrix is not None:
            self.encoding_matrix = cached_encoding_matrix
        else:
            self.batch_encode(self.corpus, True, show_progress)
            self.encoding_matrix = torch.stack([sample["full_encoding"] for sample in self.corpus]).to(device)
        self.emb_size = self.encoding_matrix.shape[1]

    def compute_encodings(self, cached_encoding_matrix: torch.Tensor = None):
        print("Encoding samples...")
        with torch.no_grad():
            self.batch_encode(self.data, False)
            if cached_encoding_matrix is not None:
                self.encoding_matrix = cached_encoding_matrix
            else:
                if self.options.sm == SamplingMethod.SIMILARITY.value:
                    if id(self.corpus) != id(self.data): # No need to re-encode context if corpus is same as data
                        self.batch_encode(self.corpus, False)
                    self.encoding_matrix = torch.stack([sample["context_encoding"] for sample in self.corpus]).to(device)
                else:
                    self.batch_encode(self.corpus, True)
                    self.encoding_matrix = torch.stack([sample["full_encoding"] for sample in self.corpus]).to(device)
        self.emb_size = self.encoding_matrix.shape[1]

        # Construct index for sample lookup
        if self.options.sm == SamplingMethod.SIMILARITY.value:
            encoding_matrix_np = self.encoding_matrix.detach().cpu().numpy()
            self.index = faiss.IndexFlatL2(self.emb_size)
            self.index.add(encoding_matrix_np)

    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_greedy(self, greedy: bool):
        self.greedy = greedy

    def get_random_examples(self, index: int, corp_eq_data: bool):
        random_example_idxs = np.array(random.sample(range(len(self.corpus)), self.options.num_examples + 1))
        if corp_eq_data:
            random_example_idxs = random_example_idxs[random_example_idxs != index]
        random_example_idxs = random_example_idxs[:self.options.num_examples]
        return random_example_idxs, None

    def get_knn_examples(self, index: int, cur_sample: DataSample, corp_eq_data: bool):
        top_neighbor_indices = self.index.search(
            cur_sample["context_encoding"].unsqueeze(0).cpu().numpy(), self.options.num_examples + 1)[1][0]
        if corp_eq_data:
            top_neighbor_indices = top_neighbor_indices[top_neighbor_indices != index]
        top_neighbor_indices = top_neighbor_indices[:self.options.num_examples]
        top_neighbor_indices = np.flip(top_neighbor_indices).copy()
        return top_neighbor_indices, None

    def get_policy_sampled_examples(self, index: int, cur_sample: DataSample, corp_eq_data: bool):
        example_idxs: List[int] = []
        used_idxs: List[int] = []
        if corp_eq_data:
            used_idxs.append(index)
        example_encodings = torch.empty((0, self.emb_size)).to(device)
        random_example_idxs, _ = self.get_random_examples(index, corp_eq_data)
        while not example_idxs or example_idxs[-1] != self.eos_idx:
            activations = self.retriever.get_activations(
                current_sample_encoding=cur_sample["context_encoding"],
                example_encodings=example_encodings,
                all_example_encodings=self.encoding_matrix,
                used_idxs=used_idxs
            )
            if self.options.sm == SamplingMethod.EPSILON_GREEDY.value:
                # If roll is above epsilon then sample from retriever, otherwise pick random sample
                if random.random() > self.epsilon:
                    example_idx = torch.argmax(activations).item()
                else:
                    # TODO: should use activations to account for used_idxs and eos
                    example_idx = random_example_idxs[0]
            elif self.options.sm == SamplingMethod.SOFTMAX.value:
                # Sample from policy at current state
                if self.options.top_k:
                    _, top_k_act_idxs = torch.topk(activations, self.options.top_k)
                    new_activations = torch.full_like(activations, -torch.inf)
                    new_activations[top_k_act_idxs] = activations[top_k_act_idxs]
                    activations = new_activations
                pi_cur = torch.softmax(activations, dim=0)
                example_idx = torch.multinomial(pi_cur, 1).item()
            used_idxs.append(example_idx)
            random_example_idxs = random_example_idxs[random_example_idxs != example_idx]
            example_idxs.append(example_idx)
            if example_idx == self.eos_idx:
                example_encoding = torch.zeros((self.emb_size,))
            else:
                example_encoding = self.corpus[example_idx]["full_encoding"]
            example_encodings = torch.cat([example_encodings, example_encoding.unsqueeze(0).to(device)], dim=0)
        return example_idxs, example_encodings

    def get_beam_search_examples(self, index: int, cur_sample: DataSample, corp_eq_data: bool):
        beams = [{
            "example_idxs": [],
            "used_idxs": [index] if corp_eq_data else [],
            "example_encodings": torch.empty((0, self.emb_size)).to(device),
            "prob": 1.0
        }]
        while not all(beam["example_idxs"] and beam["example_idxs"][-1] == self.eos_idx for beam in beams):
            beam_cands = []
            for beam in beams:
                if beam["example_idxs"] and beam["example_idxs"][-1] == self.eos_idx:
                    beam_cands.append(beam)
                    continue
                if self.options.sm == SamplingMethod.VF.value:
                    # TODO: if this is really slow then can cache previous states and pass to initial hidden state
                    activations = self.retriever.get_last_vfe(
                        current_sample_encodings=cur_sample["context_encoding"].repeat(self.encoding_matrix.shape[0], 1),
                        example_encodings=torch.cat([
                            beam["example_encodings"].unsqueeze(0).repeat(self.encoding_matrix.shape[0], 1, 1),
                            self.encoding_matrix.unsqueeze(1)
                        ], dim=1),
                    )
                    activations[beam["used_idxs"]] = -torch.inf
                else:
                    activations = self.retriever.get_activations(
                        current_sample_encoding=cur_sample["context_encoding"],
                        example_encodings=beam["example_encodings"],
                        all_example_encodings=self.encoding_matrix,
                        used_idxs=beam["used_idxs"]
                    )
                pi_cur = torch.softmax(activations, dim=0)
                if len(beam["example_idxs"]) < self.options.num_examples:
                    example_idx_cands = torch.topk(activations, self.options.beam_width)[1].tolist()
                else:
                    example_idx_cands = [self.eos_idx]
                for example_idx in example_idx_cands:
                    if example_idx == self.eos_idx:
                        # TODO: if we want to do early stopping with baseline, need to use learnable eos token embedding
                        example_encoding = torch.zeros((self.emb_size,))
                    else:
                        example_encoding = self.corpus[example_idx]["full_encoding"]
                    new_example_encodings = torch.cat([
                        beam["example_encodings"],
                        example_encoding.unsqueeze(0).to(device)
                    ], dim=0)
                    beam_cands.append({
                        "example_idxs": beam["example_idxs"] + [example_idx],
                        "used_idxs": beam["used_idxs"] + [example_idx],
                        "example_encodings": new_example_encodings,
                        "prob": pi_cur[example_idx].item() * (1 if self.options.sm == SamplingMethod.VF.value else beam["prob"])
                    })
            beams = sorted(beam_cands, key=lambda beam: -beam["prob"])[:self.options.beam_width]
        top_beam = sorted(beam_cands, key=lambda beam: -beam["prob"])[0]
        return top_beam["example_idxs"], top_beam["example_encodings"]

    def __getitem__(self, index: int) -> ICLSample:
        # Set retriever to eval mode if using it
        training = self.retriever and self.retriever.training
        if training:
            self.retriever.eval()

        # Get current sample
        cur_sample = self.data[index]

        # Re-compute current sample encoding if using trainable encoder
        if self.trainable_encoder:
            self.batch_encode([cur_sample], False, False)

        # Don't resample current sample if corpus and training set are the same
        corp_eq_data = id(self.corpus) == id(self.data)

        # Get examples based on sampling method
        with torch.no_grad():
            if self.options.sm == SamplingMethod.RANDOM.value:
                example_idxs, example_encodings = self.get_random_examples(index, corp_eq_data)
            elif self.options.sm == SamplingMethod.SIMILARITY.value:
                example_idxs, example_encodings = self.get_knn_examples(index, cur_sample, corp_eq_data)
            elif self.options.sm == SamplingMethod.COMPLEX.value:
                example_idxs, example_encodings = self.complex_example_idxs, None
            else:
                if self.greedy:
                    example_idxs, example_encodings = self.get_beam_search_examples(index, cur_sample, corp_eq_data)
                else:
                    example_idxs, example_encodings = self.get_policy_sampled_examples(index, cur_sample, corp_eq_data)

        # Construct prompt
        examples = [self.corpus[example_idx] for example_idx in example_idxs if example_idx != self.eos_idx]
        prompt = ""
        if self.prompt_prefix:
            prompt += self.prompt_prefix + "\n\n"
        prompt += "\n\n".join([example["lm_context"] + example["lm_label"] for example in examples])
        prompt += "\n\n" + cur_sample["lm_context"]

        # Set retriever back to training mode if necessary
        if training:
            self.retriever.train()

        if corp_eq_data:
            assert not any([id(ex) == id(cur_sample) for ex in examples])

        return {
            "prompt": prompt,
            "label": cur_sample["lm_label"],
            "meta_data": cur_sample["meta_data"],
            "current_sample_encoding": cur_sample.get("context_encoding"),
            "example_encodings": example_encodings,
            "policy_example_indices": example_idxs,
            "all_example_encodings": self.retriever and self.encoding_matrix,
        }

    def __len__(self):
        return len(self.data)

class Collator():
    def __init__(self, eos_idx: int):
        self.eos_idx = eos_idx

    def __call__(self, batch: List[ICLSample]) -> CollatedBatch:
        return {
            "prompts": [sample["prompt"] for sample in batch],
            "labels": [sample["label"] for sample in batch],
            "outputs": [sample["output"] for sample in batch] if batch[0].get("output") else None,
            "meta_data": [sample["meta_data"] for sample in batch],
            "current_sample_encodings": torch.stack(
                [sample["current_sample_encoding"] for sample in batch]
            ).to(device) if batch[0]["current_sample_encoding"] is not None else None,
            "example_encodings": pad_sequence(
                [sample["example_encodings"] for sample in batch],
                batch_first=True
            ).to(device) if batch[0]["example_encodings"] is not None else None,
            "seq_len": torch.tensor([
                len(sample["example_encodings"]) for sample in batch
            ]) if batch[0]["example_encodings"] is not None else None,
            "policy_example_indices": pad_sequence(
                [torch.LongTensor(sample["policy_example_indices"]) for sample in batch],
                batch_first=True,
                padding_value=self.eos_idx
            ).to(device),
            "all_example_encodings": batch[0]["all_example_encodings"],
        }
