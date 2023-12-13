from typing import TypedDict, List
import torch

from reticl.data_loading.data_types import DatasetConfig
from reticl.data_loading.reticl_dataset import ICLSample, RetICLDataset
from reticl.models.retriever import Retriever
from reticl.models.generator import GeneratorResult
from reticl.utils import TrainOptions, device

class PreloadedSample(TypedDict):
    prompt: str
    label: str
    output: GeneratorResult
    input_metadata: dict
    example_metadatas: List[dict]
    input_index: int
    policy_example_indices: List[int]

class PretrainDataset(RetICLDataset):
    def __init__(self, preloaded_samples: List[PreloadedSample], dataset_config: DatasetConfig,
                 split: str, retriever: Retriever, options: TrainOptions, compute_initial_encodings: bool):
        super().__init__(dataset_config, split, retriever, options, compute_initial_encodings)
        self.preloaded_samples = preloaded_samples
        correct = sum(dataset_config["check_correct"](sample["input_metadata"], sample["output"]["text"]) for sample in self.preloaded_samples)
        print(f"Correct: {correct}/{len(self.preloaded_samples)} ({correct / len(self.preloaded_samples) * 100:.2f}%)")

    def __len__(self):
        return len(self.preloaded_samples)

    def __getitem__(self, index: int) -> ICLSample:
        cur_sample = self.preloaded_samples[index]
        og_sample = self.data[cur_sample["input_index"]]

        # Re-compute current sample encoding if using trainable encoder
        if self.trainable_encoder:
            self.batch_encode([og_sample], False, False)

        return {
            "prompt": cur_sample["prompt"],
            "label": cur_sample["label"],
            "output": cur_sample["output"],
            "meta_data": cur_sample["input_metadata"],
            "current_sample_encoding": og_sample["context_encoding"],
            "example_encodings": torch.stack([
                self.corpus[example_idx]["full_encoding"]
                if example_idx != self.eos_idx else
                torch.zeros((self.emb_size,)).to(device)
                for example_idx in cur_sample["policy_example_indices"]
            ]),
            "all_example_encodings": None,
            "policy_example_indices": cur_sample["policy_example_indices"]
        }
