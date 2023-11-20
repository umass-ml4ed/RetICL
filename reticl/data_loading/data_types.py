from typing import TypedDict, List, Optional, Callable, Tuple, Union
import torch

from reticl.utils import TrainOptions

class DataSample(TypedDict):
    lm_context: str
    lm_label: str
    encoder_context: str
    encoder_label: str
    meta_data: dict
    context_encoding: torch.Tensor
    full_encoding: torch.Tensor

GetDataFunction = Callable[[str, TrainOptions], Tuple[List[dict], Optional[List[dict]]]]
ProcessDataFunction = Callable[[dict], DataSample]
CheckCorrectFunction = Callable[[dict, str], Union[bool, float]]
CheckCorrectBatchFunction = Callable[[List[dict], List[str]], torch.Tensor]
ComplexityMetric = Callable[[DataSample], int]

class DatasetConfig(TypedDict):
    get_data: GetDataFunction
    process_sample: ProcessDataFunction
    check_correct: Optional[CheckCorrectFunction]
    check_correct_batch: Optional[CheckCorrectBatchFunction]
    complexity_metric: Optional[ComplexityMetric]
    prompt_prefix: Optional[str]
