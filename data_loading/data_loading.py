from typing import TypedDict, List, Optional
from abc import abstractmethod
import random
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
import faiss

from models.retriever import Retriever
from constants import SamplingMethod
from utils import device, TrainOptions

class DataSample(TypedDict):
    context: str
    label: str
    meta_data: dict
    context_encoding: Optional[torch.Tensor]
    full_encoding: Optional[torch.Tensor]

class ICLSample(TypedDict):
    # For evaluation
    prompt: str
    label: str
    meta_data: dict
    # For retriever
    current_sample_encoding: torch.Tensor
    example_encodings: torch.Tensor
    # For PG version of model
    top_k_example_encodings: torch.Tensor
    policy_example_indices: torch.Tensor

class CollatedBatch(TypedDict):
    # For evaluation
    prompts: List[str]
    labels: List[str]
    meta_data: List[dict]
    # For retriever
    current_sample_encodings: torch.Tensor
    example_encodings: torch.Tensor
    num_examples: torch.Tensor
    # For PG version of model
    top_k_example_encodings: torch.Tensor
    policy_example_indices: torch.Tensor

class DatasetBase(TorchDataset):
    def __init__(self, samples: list, corpus: Optional[list], retriever: Optional[Retriever], options: TrainOptions):
        super().__init__()

        # Process data samples
        self.data: List[DataSample] = [self.process_sample(sample) for sample in samples]
        if corpus:
            self.corpus: List[DataSample] = [self.process_sample(sample) for sample in corpus]
        else:
            self.corpus = self.data

        if options.method != SamplingMethod.RANDOM.value:
            # Get vector encodings for all samples
            print("Encoding samples...")
            encoder = SentenceTransformer(options.encoder_model)
            for sample in tqdm(self.data):
                sample["context_encoding"] = encoder.encode(sample["context"], convert_to_tensor=True)
            if options.method == SamplingMethod.SIMILARITY.value:
                if corpus: # No need to re-encode context if corpus is same as data
                    for sample in tqdm(self.corpus):
                        sample["context_encoding"] = encoder.encode(sample["context"], convert_to_tensor=True)
                self.encoding_matrix = torch.stack([sample["context_encoding"] for sample in self.corpus])
            else:
                for sample in tqdm(self.corpus):
                    sample["full_encoding"] = encoder.encode(sample["context"] + sample["label"], convert_to_tensor=True)
                self.encoding_matrix = torch.stack([sample["full_encoding"] for sample in self.corpus])

            # Construct index for sample lookup
            self.emb_size = self.encoding_matrix.shape[1]
            encoding_matrix_np = self.encoding_matrix.cpu().numpy()
            self.index = faiss.IndexFlatL2(self.emb_size)
            # quantizer = faiss.IndexFlatL2(self.emb_size)
            # self.index = faiss.IndexIVFFlat(quantizer, self.emb_size, 100)
            # self.index.nprobe = 10
            # self.index.train(encoding_matrix_np)
            self.index.add(encoding_matrix_np)

        self.options = options
        self.retriever = retriever
        self.epsilon = 0.0
        self.greedy = False

    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_greedy(self, greedy: bool):
        self.greedy = greedy

    @abstractmethod
    def process_sample(self, sample) -> DataSample:
        # Method should be overriden by child class
        return sample

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

            # Initialize context
            prompt = ""
            examples: List[DataSample] = []
            used_idxs: List[int] = []
            # Either sampling from original dataset or separate corpus
            corp_eq_data = id(self.corpus) == id(self.data)
            if corp_eq_data:
                used_idxs.append(index)
            # Group of examples to draw from when doing random sampling
            random_example_idxs = np.array(random.sample(range(len(self.corpus)), self.options.num_examples + 1))
            if corp_eq_data:
                random_example_idxs = random_example_idxs[random_example_idxs != index]
            # Group of examples to draw from when doing similarity sampling
            if self.options.method == SamplingMethod.SIMILARITY.value:
                top_neighbor_indices = self.index.search(
                    cur_sample["context_encoding"].unsqueeze(0).cpu().numpy(), self.options.num_examples + 1)[1][0]
                if corp_eq_data:
                    top_neighbor_indices = top_neighbor_indices[top_neighbor_indices != index]
            # Keep track of example encodings for retriever
            if self.options.method in (SamplingMethod.RANDOM.value, SamplingMethod.SIMILARITY.value):
                example_encodings = None
            else:
                example_encodings = torch.empty((0, self.emb_size)).to(device)
            sample_from_policy = self.options.method in (SamplingMethod.PG.value, SamplingMethod.RWB.value, SamplingMethod.PPO.value)
            if sample_from_policy:
                k = self.options.top_k or len(self.corpus) - (1 if corp_eq_data else 0)
                top_k_example_encodings = torch.empty((0, k, self.emb_size)).to(device)
                policy_example_indices: List[int] = []
            else:
                top_k_example_encodings = None
                policy_example_indices = None

            # Retrieve examples until context is full
            while len(examples) < self.options.num_examples:
                if self.options.method == SamplingMethod.MCC.value:
                    # Epsilon-greedy: if roll is above epsilon then sample from retriever, otherwise pick random sample
                    if self.greedy or random.random() > self.epsilon:
                        qv = self.retriever.get_query_vector(
                            current_sample_encoding=cur_sample["context_encoding"],
                            example_encodings=example_encodings,
                        )
                        example_idx = self._get_top_unused_example(qv, used_idxs)
                    else:
                        example_idx = random_example_idxs[0]
                elif sample_from_policy:
                    # Policy Gradient: sample from approximate policy at current state
                    qv = self.retriever.get_query_vector(
                        current_sample_encoding=cur_sample["context_encoding"],
                        example_encodings=example_encodings,
                    )
                    if self.greedy:
                        example_idx = self._get_top_unused_example(qv, used_idxs)
                    else:
                        # Get set of examples to sample from; either top k or full corpus
                        if self.options.top_k:
                            top_k_indices = self.index.search(
                                qv.unsqueeze(0).cpu().numpy(),
                                self.options.top_k
                            )[1][0]
                        else:
                            top_k_indices = np.arange(len(self.corpus))
                        # Randomly sample an example from the policy
                        top_k_vecs = self.encoding_matrix[top_k_indices]
                        activations = torch.matmul(qv, top_k_vecs.T)
                        activations[used_idxs] = -torch.inf
                        pi_cur = torch.softmax(activations, dim=0)
                        local_example_idx = torch.multinomial(pi_cur, 1)
                        # Add example and distribution to running lists
                        top_k_example_encodings = torch.cat([top_k_example_encodings, top_k_vecs.unsqueeze(0)], dim=0)
                        policy_example_indices.append(local_example_idx)
                        example_idx = top_k_indices[local_example_idx]
                elif self.options.method == SamplingMethod.RANDOM.value:
                    example_idx = random_example_idxs[0]
                elif self.options.method == SamplingMethod.SIMILARITY.value:
                    example_idx = top_neighbor_indices[len(examples)]

                # Exclude current example from future selections
                used_idxs.append(example_idx)
                random_example_idxs = random_example_idxs[random_example_idxs != example_idx]

                # Add retrieved example to the context
                example = self.corpus[example_idx]
                examples.append(example)
                prompt += example["context"] + example["label"] + "\n\n"
                if self.options.method not in (SamplingMethod.RANDOM.value, SamplingMethod.SIMILARITY.value):
                    example_encoding = example["full_encoding"]
                    example_encodings = torch.cat([example_encodings, example_encoding.unsqueeze(0).to(device)], dim=0)

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
            "top_k_example_encodings": top_k_example_encodings,
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
            ).to(device) if batch[0]["policy_example_indices"] is not None else None,
            "top_k_example_encodings": pad_sequence(
                [sample["top_k_example_encodings"] for sample in batch],
                batch_first=True
            ).to(device) if batch[0]["top_k_example_encodings"] is not None else None,
        }
