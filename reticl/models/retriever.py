from typing import Union

from reticl.models.reticl_rnn import RetICLRNN
from reticl.models.reticl_attn import RetICLAttn
from reticl.models.reticl_ind import RetICLInd
from reticl.constants import ModelType
from reticl.utils import TrainOptions, device

Retriever = Union[RetICLRNN, RetICLAttn, RetICLInd]

def retriever_model(options: TrainOptions, use_bias: bool = False, mask_prev_examples: bool = True, num_critics: int = 0) -> Retriever:
    if options.model_type == ModelType.RNN.value or options.model_type.startswith(ModelType.LSTM.value):
        return RetICLRNN(options, use_bias, mask_prev_examples, num_critics).to(device)
    if options.model_type == ModelType.ATTN.value:
        return RetICLAttn(options, use_bias, mask_prev_examples, num_critics).to(device)
    if options.model_type == ModelType.IND.value:
        return RetICLInd(options, use_bias, mask_prev_examples, num_critics).to(device)
    raise Exception(f"Model type {options.model_type} not supported!")
