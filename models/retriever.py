from typing import Union

from models.reticl_rnn import RetICLRNN
from models.reticl_attn import RetICLAttn
from models.reticl_ind import RetICLInd
from constants import ModelType
from utils import TrainOptions, device

Retriever = Union[RetICLRNN, RetICLAttn, RetICLInd]

def retriever_model(options: TrainOptions, use_bias: bool = False) -> Retriever:
    if options.model_type in (ModelType.RNN.value, ModelType.LSTM.value):
        return RetICLRNN(options, use_bias).to(device)
    if options.model_type == ModelType.ATTN.value:
        return RetICLAttn(options, use_bias).to(device)
    if options.model_type == ModelType.IND.value:
        return RetICLInd(options, use_bias).to(device)
    raise Exception(f"Model type {options.model_type} not supported!")
