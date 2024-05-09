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
    # For rewards
    sub_prompts: List[str]
    example_similarity: torch.Tensor
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
    # For rewards
    sub_prompts: List[List[str]]
    example_similarity: torch.Tensor
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

        if options.int_reward_sim:
            print("Computing example similarity...")
            if self.trainable_encoder:
                similarity_encoder = SentenceTransformer(self.options.encoder_model or "all-distilroberta-v1")
            else:
                similarity_encoder = self.encoder
            with torch.no_grad():
                example_encodings = self._get_corpus_encodings(False, show_progress=True, encoder=similarity_encoder)
            if id(self.corpus) == id(self.data):
                self.example_sim_matrix = torch.matmul(example_encodings, example_encodings.T)
                # Prevent calculating max similarity with self
                self.example_sim_matrix[torch.eye(self.example_sim_matrix.shape[0]).bool()] = 0
            else:
                dataset_encodings = torch.stack([sample["context_encoding"] for sample in self.data]).to(device)
                self.example_sim_matrix = torch.matmul(dataset_encodings, example_encodings.T)
            max_sim = self.example_sim_matrix.max(dim=1, keepdim=True).values
            min_sim = self.example_sim_matrix.min(dim=1, keepdim=True).values
            self.example_sim_matrix = (self.example_sim_matrix - min_sim) / (max_sim - min_sim)

        if options.sm == SamplingMethod.COMPLEX.value:
            complexity_metric = dataset_config.get("complexity_metric", lambda sample: sample["lm_label"].count("\n"))
            corpus_complexity = np.array([-complexity_metric(sample) for sample in self.corpus])
            sorted_idxs = np.argsort(corpus_complexity)
            min_complexity = corpus_complexity[sorted_idxs[options.num_examples]]
            self.complex_example_idxs = (corpus_complexity <= min_complexity).nonzero()[0].tolist()

    def batch_encode(self, samples: List[DataSample], inc_label: bool, show_progress: bool = True, encoder = None):
        encoder = encoder or self.encoder
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

    def _get_corpus_encodings(self, inc_label: bool, show_progress: bool, encoder = None):
        if inc_label:
            self.batch_encode(self.corpus, True, show_progress=show_progress, encoder=encoder)
            return torch.stack([sample["full_encoding"] for sample in self.corpus]).to(device)
        if id(self.corpus) != id(self.data): # No need to re-encode context if corpus is same as data
            self.batch_encode(self.corpus, False, show_progress=show_progress, encoder=encoder)
        return torch.stack([sample["context_encoding"] for sample in self.corpus]).to(device)

    def compute_corpus_encodings(self, inc_label: bool = True, show_progress: bool = True,
                                 cached_encoding_matrix: torch.Tensor = None):
        if cached_encoding_matrix is not None:
            self.encoding_matrix = cached_encoding_matrix
        else:
            self.encoding_matrix = self._get_corpus_encodings(inc_label, show_progress)
        self.emb_size = self.encoding_matrix.shape[1]

    def compute_encodings(self, cached_encoding_matrix: torch.Tensor = None):
        print("Encoding samples...")
        with torch.no_grad():
            self.batch_encode(self.data, False, show_progress=True)
            self.compute_corpus_encodings(
                inc_label=self.options.sm != SamplingMethod.SIMILARITY.value,
                cached_encoding_matrix=cached_encoding_matrix)

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
        random_example_idxs = np.concatenate([random_example_idxs, [self.eos_idx]])
        return random_example_idxs, None

    def get_knn_examples(self, index: int, cur_sample: DataSample, corp_eq_data: bool):
        top_neighbor_indices = self.index.search(
            cur_sample["context_encoding"].unsqueeze(0).cpu().numpy(), self.options.num_examples + 1)[1][0]
        if corp_eq_data:
            top_neighbor_indices = top_neighbor_indices[top_neighbor_indices != index]
        top_neighbor_indices = top_neighbor_indices[:self.options.num_examples]
        top_neighbor_indices = np.flip(top_neighbor_indices).copy()
        top_neighbor_indices = np.concatenate([top_neighbor_indices, [self.eos_idx]])
        return top_neighbor_indices, None

    def get_complexity_examples(self, index: int, corp_eq_data: bool):
        example_idxs = np.array(random.sample(self.complex_example_idxs, self.options.num_examples + 1))
        if corp_eq_data:
            example_idxs = example_idxs[example_idxs != index]
        example_idxs = example_idxs[:self.options.num_examples]
        example_idxs = np.concatenate([example_idxs, [self.eos_idx]])
        return example_idxs, None

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
            "score": 1.0,
            "prob": 1.0
        }]
        # Keep going until all beams end with eos
        while not all(beam["example_idxs"] and beam["example_idxs"][-1] == self.eos_idx for beam in beams):
            beam_cands = []
            for beam in beams:
                if beam["example_idxs"] and beam["example_idxs"][-1] == self.eos_idx:
                    beam_cands.append(beam)
                    continue
                if len(beam["example_idxs"]) < self.options.num_examples:
                    if self.options.sm == SamplingMethod.VF.value:
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
                    example_idx_cands = torch.topk(activations, self.options.beam_width)[1].tolist()
                    scores = self.retriever.get_last_vfe(
                        current_sample_encodings=cur_sample["context_encoding"].repeat(self.options.beam_width, 1),
                        example_encodings=torch.cat([
                            beam["example_encodings"].unsqueeze(0).repeat(self.options.beam_width, 1, 1),
                            self.encoding_matrix[example_idx_cands].unsqueeze(1)
                        ], dim=1)
                    )
                    pi_cur = torch.softmax(activations, dim=0)
                    probs = pi_cur[example_idx_cands].tolist()
                else:
                    example_idx_cands = [self.eos_idx]
                    probs = [1.0]
                for cand_idx, example_idx in enumerate(example_idx_cands):
                    prob = probs[cand_idx] * beam["prob"]
                    if example_idx == self.eos_idx:
                        # TODO: if we want to do early stopping with vf, need to use learnable eos token embedding
                        example_encoding = torch.zeros((self.emb_size,)).to(device)
                        # score = beam["score"] if self.options.sm == SamplingMethod.VF.value else 1
                        score = beam["score"] # Keep score from last example
                    else:
                        example_encoding = self.corpus[example_idx]["full_encoding"]
                        score = scores[cand_idx]
                    new_example_encodings = torch.cat([
                        beam["example_encodings"],
                        example_encoding.unsqueeze(0).to(device)
                    ], dim=0)
                    beam_cands.append({
                        "example_idxs": beam["example_idxs"] + [example_idx],
                        "used_idxs": beam["used_idxs"] + [example_idx],
                        "example_encodings": new_example_encodings,
                        # "score": score * (1 if self.options.sm == SamplingMethod.VF.value else beam["score"])
                        "score": score,
                        "prob": prob
                    })
            beams = sorted(beam_cands, key=lambda beam: -beam["score"])[:self.options.beam_width]
        top_beam = sorted(beams, key=lambda beam: -beam["score"])[0]
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
            self.batch_encode([cur_sample], False, show_progress=False)

        # Don't resample current sample if corpus and training set are the same
        corp_eq_data = id(self.corpus) == id(self.data)

        # Get examples based on sampling method
        with torch.no_grad():
            if self.options.sm == SamplingMethod.RANDOM.value:
                example_idxs, example_encodings = self.get_random_examples(index, corp_eq_data)
            elif self.options.sm == SamplingMethod.SIMILARITY.value:
                example_idxs, example_encodings = self.get_knn_examples(index, cur_sample, corp_eq_data)
            elif self.options.sm == SamplingMethod.COMPLEX.value:
                example_idxs, example_encodings = self.get_complexity_examples(index, corp_eq_data)
            else:
                if self.greedy:
                    example_idxs, example_encodings = self.get_beam_search_examples(index, cur_sample, corp_eq_data)
                else:
                    example_idxs, example_encodings = self.get_policy_sampled_examples(index, cur_sample, corp_eq_data)

        if self.options.int_reward_sim:
            example_similarity = torch.stack([
                self.example_sim_matrix[index, example_idx] for example_idx in example_idxs
                if example_idx != self.eos_idx
            ])
        else:
            example_similarity = None

        # Construct prompt
        examples = [self.corpus[example_idx] for example_idx in example_idxs if example_idx != self.eos_idx]
        prompt = ""
        sub_prompts = []
        if self.prompt_prefix:
            prompt += self.prompt_prefix + "\n\n"
        for example in examples:
            prompt += example["lm_context"] + example["lm_label"] + "\n\n"
            sub_prompts.append(prompt + cur_sample["lm_context"])
        prompt += cur_sample["lm_context"]

        # Set retriever back to training mode if necessary
        if training:
            self.retriever.train()

        if corp_eq_data:
            assert not any([id(ex) == id(cur_sample) for ex in examples])

        return {
            "prompt": prompt,
            "label": cur_sample["lm_label"],
            "meta_data": cur_sample["meta_data"],
            "sub_prompts": sub_prompts,
            "example_similarity": example_similarity,
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
            "sub_prompts": [sample["sub_prompts"] for sample in batch],
            "example_similarity": pad_sequence(
                [sample["example_similarity"] for sample in batch],
                batch_first=True
            ).to(device) if batch[0]["example_similarity"] is not None else None,
            "current_sample_encodings": torch.stack(
                [sample["current_sample_encoding"] for sample in batch]
            ).to(device) if batch[0]["current_sample_encoding"] is not None else None,
            "example_encodings": pad_sequence(
                [sample["example_encodings"] for sample in batch],
                batch_first=True
            ).to(device) if batch[0]["example_encodings"] is not None else None,
            "seq_len": torch.tensor([
                len(sample["policy_example_indices"]) for sample in batch
            ]),
            "policy_example_indices": pad_sequence(
                [torch.LongTensor(sample["policy_example_indices"]) for sample in batch],
                batch_first=True,
                padding_value=self.eos_idx
            ).to(device),
            "all_example_encodings": batch[0]["all_example_encodings"],
        }

def filter_batch(batch: CollatedBatch, mask: torch.Tensor):
    for key, value in batch.items():
        if type(value) is list:
            batch[key] = [value[i] for i, include in enumerate(mask) if include]
        elif type(value) is torch.Tensor and key != "all_example_encodings":
            batch[key] = value[mask]
    return batch
