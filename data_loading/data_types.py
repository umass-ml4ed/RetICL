from typing import TypedDict, List, Optional, Callable, Tuple
import torch

from utils import TrainOptions

class DataSample(TypedDict):
    context: str
    label: str
    meta_data: dict
    context_encoding: Optional[torch.Tensor]
    full_encoding: Optional[torch.Tensor]

GetDataFunction = Callable[[str, TrainOptions], Tuple[List[dict], Optional[List[dict]]]]
ProcessDataFunction = Callable[[dict], DataSample]
