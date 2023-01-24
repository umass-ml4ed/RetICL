from typing import Union

from models.reticl import RetICL
from models.ret_ind import RetInd
from utils import TrainOptions, device

Retriever = Union[RetICL, RetInd]

def retriever_model(options: TrainOptions):
    if options.baseline:
        return RetInd(options).to(device)
    return RetICL(options).to(device)
