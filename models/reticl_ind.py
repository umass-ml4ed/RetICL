import torch
from torch import nn

from models.reticl_base import RetICLBase, orthogonal_init_
from utils import TrainOptions
from constants import Init

class RetICLInd(RetICLBase):
    def __init__(self, options: TrainOptions):
        super().__init__(options)
        self.h_0_transform = nn.Sequential(
            nn.Linear(self.emb_size, options.hidden_size),
            nn.Tanh()
        )
        if options.init == Init.ORTHOGONAL.value:
            orthogonal_init_(self.h_0_transform)

    def get_latent_states(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor, **kwargs):
        batch_size, num_examples = example_encodings.shape[:2]
        # Independent model has same latent state for all examples
        h_0 = self.h_0_transform(current_sample_encodings) # (N x H)
        latent_states = h_0.repeat(1, num_examples).view(batch_size, num_examples, -1) # (N x L x H)
        return latent_states

    def get_query_vector(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor):
        h_0 = self.h_0_transform(current_sample_encoding)
        return torch.matmul(h_0, self.bilinear)
