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
from constants import SamplingMethod, EncoderModelType
from utils import device, TrainOptions, is_pg

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
    all_example_encodings: Optional[torch.Tensor]

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
    all_example_encodings: Optional[torch.Tensor]

class RetICLDataset(TorchDataset):
    def __init__(self, get_data: GetDataFunction, process_sample: ProcessDataFunction,
                 split: str, retriever: Optional[Retriever], options: TrainOptions):
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
        if options.sm != SamplingMethod.RANDOM.value:
            self.compute_encodings()

    def compute_encodings(self):
        print("Encoding samples...")
        if self.options.encoder_model_type == EncoderModelType.BERT.value:
            encoder = BertModel.from_pretrained(self.options.encoder_model or "bert-base-cased").to(device)
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        elif self.options.encoder_model_type == EncoderModelType.GPT2.value:
            encoder = GPT2Model.from_pretrained(self.options.encoder_model or "gpt2").to(device)
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            encoder = SentenceTransformer(self.options.encoder_model or "all-mpnet-base-v2")

        def batch_encode(samples: List[DataSample], inc_label: bool):
            batch_size = 10
            for batch_start_idx in tqdm(range(0, len(samples), batch_size)):
                batch = samples[batch_start_idx : batch_start_idx + batch_size]
                if self.options.encoder_model_type == EncoderModelType.SBERT.value:
                    seq_strings = [
                        (sample["context"] + sample["label"]) if inc_label else sample["context"]
                        for sample in batch
                    ]
                    outputs = encoder.encode(seq_strings, convert_to_tensor=True)
                elif self.options.encoder_model_type == EncoderModelType.BERT.value:
                    inputs = tokenizer(
                        [sample["context"] for sample in batch],
                        [sample["label"] for sample in batch] if inc_label else None,
                        return_tensors="pt", padding=True, truncation=True, max_length=512
                    ).to(device)
                    outputs = encoder(**inputs).pooler_output
                    # outputs = encoder(**inputs).last_hidden_state[:, 0]
                elif self.options.encoder_model_type == EncoderModelType.GPT2.value:
                    seq_strings = [
                        (sample["context"] + sample["label"]) if inc_label else sample["context"]
                        for sample in batch
                    ]
                    inputs = tokenizer(seq_strings, return_tensors="pt", padding=True).to(device)
                    outputs = encoder(**inputs).last_hidden_state[
                        torch.arange(inputs.input_ids.shape[0]),
                        torch.sum(inputs.attention_mask, dim=-1) - 1
                    ]
                for sample, encoding in zip(batch, outputs):
                    sample["full_encoding" if inc_label else "context_encoding"] = encoding

        with torch.no_grad():
            batch_encode(self.data, False)
            if self.options.sm == SamplingMethod.SIMILARITY.value:
                if id(self.corpus) != id(self.data): # No need to re-encode context if corpus is same as data
                    batch_encode(self.corpus, False)
                self.encoding_matrix = torch.stack([sample["context_encoding"] for sample in self.corpus])
            else:
                batch_encode(self.corpus, True)
                self.encoding_matrix = torch.stack([sample["full_encoding"] for sample in self.corpus])

        # Construct index for sample lookup
        self.emb_size = self.encoding_matrix.shape[1]
        encoding_matrix_np = self.encoding_matrix.cpu().numpy()
        self.index = faiss.IndexFlatL2(self.emb_size)
        self.index.add(encoding_matrix_np)

    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_greedy(self, greedy: bool):
        self.greedy = greedy

    def _get_top_unused_example(self, qv: torch.Tensor, used_idxs: List[int]):
        top_examples = self.index.search(
            qv.unsqueeze(0).cpu().numpy(),
            1 + len(used_idxs)
        )[1][0]
        for used_idx in used_idxs:
            top_examples = top_examples[top_examples != used_idx]
        return top_examples[0]

    def __getitem__(self, index: int) -> ICLSample:
        training = self.retriever and self.retriever.training
        if training:
            self.retriever.eval()

        with torch.no_grad():
            # Get current sample
            cur_sample = self.data[index]

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
            # Keep track of example encodings for retriever
            if self.options.sm in (SamplingMethod.RANDOM.value, SamplingMethod.SIMILARITY.value):
                example_encodings = None
            else:
                example_encodings = torch.empty((0, self.emb_size)).to(device)
            if is_pg(self.options):
                all_example_encodings = torch.empty((0, len(self.corpus), self.emb_size)).to(device)
            else:
                all_example_encodings = None

            # Retrieve examples until context is full
            while len(examples) < self.options.num_examples:
                if self.options.sm == SamplingMethod.EPSILON_GREEDY.value:
                    # Epsilon-greedy: if roll is above epsilon then sample from retriever, otherwise pick random sample
                    if self.greedy or random.random() > self.epsilon:
                        qv = self.retriever.get_query_vector(
                            current_sample_encoding=cur_sample["context_encoding"],
                            example_encodings=example_encodings,
                        )
                        example_idx = self._get_top_unused_example(qv, used_idxs)
                    else:
                        example_idx = random_example_idxs[0]
                elif self.options.sm == SamplingMethod.SOFTMAX.value:
                    # Policy Gradient: sample from policy at current state
                    qv = self.retriever.get_query_vector(
                        current_sample_encoding=cur_sample["context_encoding"],
                        example_encodings=example_encodings,
                    )
                    if self.greedy:
                        example_idx = self._get_top_unused_example(qv, used_idxs)
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
                        # val_ests = self.retriever.get_all_value_estimates(
                        #     current_sample_encoding=cur_sample["context_encoding"],
                        #     all_example_encodings=self.encoding_matrix
                        # )
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
                prompt += example["context"] + example["label"] + "\n\n"
                if self.options.sm not in (SamplingMethod.RANDOM.value, SamplingMethod.SIMILARITY.value):
                    example_encoding = example["full_encoding"]
                    example_encodings = torch.cat([example_encodings, example_encoding.unsqueeze(0).to(device)], dim=0)
                if is_pg(self.options):
                    all_example_encodings = torch.cat([all_example_encodings, self.encoding_matrix.unsqueeze(0)], dim=0)

        if training:
            self.retriever.train()

        if corp_eq_data:
            assert not any([id(ex) == id(cur_sample) for ex in examples])

        return {
            "prompt": prompt + cur_sample["context"],
            "label": cur_sample["label"],
            "meta_data": cur_sample["meta_data"],
            "current_sample_encoding": cur_sample.get("context_encoding"),
            "example_encodings": example_encodings,
            "policy_example_indices": policy_example_indices,
            "all_example_encodings": all_example_encodings,
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
            "all_example_encodings": pad_sequence(
                [sample["all_example_encodings"] for sample in batch],
                batch_first=True
            ).to(device) if batch[0]["all_example_encodings"] is not None else None,
        }