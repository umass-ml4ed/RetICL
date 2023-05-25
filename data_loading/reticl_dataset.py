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

from data_loading.data_types import DataSample, GetDataFunction, ProcessDataFunction
from models.retriever import Retriever
from models.encoder import SBERTEncoder
from constants import SamplingMethod, EncoderModelType
from utils import device, TrainOptions

class ICLSample(TypedDict):
    # For evaluation
    prompt: str
    label: str
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
    meta_data: List[dict]
    # For retriever
    current_sample_encodings: torch.Tensor
    example_encodings: torch.Tensor
    policy_example_indices: torch.Tensor
    num_examples: torch.Tensor
    # For PG version of model
    all_example_encodings: torch.Tensor

class RetICLDataset(TorchDataset):
    def __init__(self, get_data: GetDataFunction, process_sample: ProcessDataFunction,
                 split: str, retriever: Optional[Retriever], options: TrainOptions, compute_intial_encodings: bool = True,
                 cached_encoding_matrix: Optional[torch.Tensor] = None):
        super().__init__()

        self.options = options
        self.retriever = retriever
        self.epsilon = 0.0
        self.greedy = False

        # Process data samples
        samples, corpus = get_data(split, options)
        self.data = [process_sample(sample) for sample in samples]
        if corpus:
            self.corpus = [process_sample(sample) for sample in corpus]
        else:
            self.corpus = self.data

        # Compute encodings
        self.trainable_encoder = False
        if options.sm != SamplingMethod.RANDOM.value:
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
                # outputs = encoder(**inputs).last_hidden_state[:, 0]
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
            encoding_matrix_np = self.encoding_matrix.cpu().numpy()
            self.index = faiss.IndexFlatL2(self.emb_size)
            self.index.add(encoding_matrix_np)

    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_greedy(self, greedy: bool):
        self.greedy = greedy

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

        with torch.no_grad():
            prompt = ""
            examples: List[DataSample] = []
            used_idxs: List[int] = []
            policy_example_indices: List[int] = []
            # Either sampling from original dataset or separate corpus
            corp_eq_data = id(self.corpus) == id(self.data)
            if corp_eq_data:
                used_idxs.append(index)
            # Group of examples to draw from when doing random sampling
            random_example_idxs = np.array(random.sample(range(len(self.corpus)), self.options.num_examples + 1))
            if corp_eq_data:
                random_example_idxs = random_example_idxs[random_example_idxs != index]
            # Group of examples to draw from when doing similarity sampling
            if self.options.sm == SamplingMethod.SIMILARITY.value:
                top_neighbor_indices = self.index.search(
                    cur_sample["context_encoding"].unsqueeze(0).cpu().numpy(), self.options.num_examples + 1)[1][0]
                if corp_eq_data:
                    top_neighbor_indices = top_neighbor_indices[top_neighbor_indices != index]
                top_neighbor_indices = top_neighbor_indices[:self.options.num_examples]
                # top_neighbor_indices = np.flip(top_neighbor_indices)
            # Keep track of example encodings for retriever
            if self.options.sm in (SamplingMethod.RANDOM.value, SamplingMethod.SIMILARITY.value):
                example_encodings = None
            else:
                example_encodings = torch.empty((0, self.emb_size)).to(device)

            # Retrieve examples until context is full
            while len(examples) < self.options.num_examples:
                if self.options.sm == SamplingMethod.EPSILON_GREEDY.value:
                    # If roll is above epsilon then sample from retriever, otherwise pick random sample
                    if self.greedy or random.random() > self.epsilon:
                        qv = self.retriever.get_query_vector(
                            current_sample_encoding=cur_sample["context_encoding"],
                            example_encodings=example_encodings,
                        )
                        activations = torch.matmul(qv, self.encoding_matrix.T)
                        activations[used_idxs] = -torch.inf
                        example_idx = torch.argmax(activations).item()
                    else:
                        example_idx = random_example_idxs[0]
                elif self.options.sm == SamplingMethod.SOFTMAX.value:
                    # Sample from policy at current state
                    qv = self.retriever.get_query_vector(
                        current_sample_encoding=cur_sample["context_encoding"],
                        example_encodings=example_encodings,
                    )
                    if self.greedy:
                        activations = torch.matmul(qv, self.encoding_matrix.T)
                        activations[used_idxs] = -torch.inf
                        example_idx = torch.argmax(activations).item()
                    else:
                        # Randomly sample an example from the policy
                        activations = torch.matmul(qv, self.encoding_matrix.T)
                        activations[used_idxs] = -torch.inf
                        if self.options.top_k:
                            _, top_k_act_idxs = torch.topk(activations, self.options.top_k)
                            new_activations = torch.full_like(activations, -torch.inf)
                            new_activations[top_k_act_idxs] = activations[top_k_act_idxs]
                            activations = new_activations
                        pi_cur = torch.softmax(activations, dim=0)
                        example_idx = torch.multinomial(pi_cur, 1).item()
                elif self.options.sm == SamplingMethod.RANDOM.value:
                    example_idx = random_example_idxs[0]
                elif self.options.sm == SamplingMethod.SIMILARITY.value:
                    example_idx = top_neighbor_indices[len(examples)]

                # Exclude current example from future selections
                used_idxs.append(example_idx)
                random_example_idxs = random_example_idxs[random_example_idxs != example_idx]

                # Add retrieved example to the context
                example = self.corpus[example_idx]
                examples.append(example)
                policy_example_indices.append(example_idx)
                prompt += example["lm_context"] + example["lm_label"] + "\n\n"
                if self.options.sm not in (SamplingMethod.RANDOM.value, SamplingMethod.SIMILARITY.value):
                    example_encoding = example["full_encoding"]
                    example_encodings = torch.cat([example_encodings, example_encoding.unsqueeze(0).to(device)], dim=0)

        # Set retriever back to training mode if necessary
        if training:
            self.retriever.train()

        if corp_eq_data:
            assert not any([id(ex) == id(cur_sample) for ex in examples])

        return {
            "prompt": prompt + cur_sample["lm_context"],
            "label": cur_sample["lm_label"],
            "meta_data": cur_sample["meta_data"],
            "current_sample_encoding": cur_sample.get("context_encoding"),
            "example_encodings": example_encodings,
            "policy_example_indices": policy_example_indices,
            "all_example_encodings": self.retriever and self.encoding_matrix,
        }

    def __len__(self):
        return len(self.data)

class Collator():
    def __call__(self, batch: List[ICLSample]) -> CollatedBatch:
        return {
            "prompts": [sample["prompt"] for sample in batch],
            "labels": [sample["label"] for sample in batch],
            "meta_data": [sample["meta_data"] for sample in batch],
            "current_sample_encodings": torch.stack(
                [sample["current_sample_encoding"] for sample in batch]
            ).to(device) if batch[0]["current_sample_encoding"] is not None else None,
            "example_encodings": pad_sequence(
                [sample["example_encodings"] for sample in batch],
                batch_first=True
            ).to(device) if batch[0]["example_encodings"] is not None else None,
            "num_examples": torch.tensor([
                len(sample["example_encodings"]) for sample in batch
            ]) if batch[0]["example_encodings"] is not None else None,
            "policy_example_indices": pad_sequence(
                [torch.LongTensor(sample["policy_example_indices"]) for sample in batch],
                batch_first=True
            ).to(device),
            "all_example_encodings": batch[0]["all_example_encodings"],
        }
