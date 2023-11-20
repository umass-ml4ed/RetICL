from typing import TypedDict, List
import torch

from reticl.data_loading.data_types import DatasetConfig
from reticl.data_loading.reticl_dataset import ICLSample, RetICLDataset
from reticl.models.generator import GeneratorResult
from reticl.utils import TrainOptions

class PreloadedSample(TypedDict):
    prompt: str
    label: str
    output: GeneratorResult
    input_metadata: dict
    example_metadatas: List[dict]
    input_index: int
    policy_example_indices: List[int]

class PretrainDataset(RetICLDataset):
    def __init__(self, preloaded_samples: List[PreloadedSample], dataset_config: DatasetConfig, split: str, options: TrainOptions):
        super().__init__(dataset_config, split, None, options)
        self.preloaded_samples = preloaded_samples

    def __len__(self):
        return len(self.preloaded_samples)

    def __getitem__(self, index: int) -> ICLSample:
        cur_sample = self.preloaded_samples[index]
        return {
            "prompt": cur_sample["prompt"],
            "label": cur_sample["label"],
            "output": cur_sample["output"],
            "meta_data": cur_sample["input_metadata"],
            "current_sample_encoding": self.data[cur_sample["input_index"]]["context_encoding"],
            "example_encodings": torch.stack([
                self.corpus[example_idx]["full_encoding"]
                for example_idx in cur_sample["policy_example_indices"]
            ]),
            "all_example_encodings": None,
            "policy_example_indices": cur_sample["policy_example_indices"]
        }
