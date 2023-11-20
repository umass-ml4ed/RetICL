import torch
from torch import nn

from reticl.models.reticl_base import RetICLBase
from reticl.utils import TrainOptions, orthogonal_init_
from reticl.constants import Init

class RetICLInd(RetICLBase):
    def __init__(self, options: TrainOptions, use_bias: bool, mask_prev_examples: bool, num_critics: int):
        super().__init__(options, use_bias, mask_prev_examples, num_critics)
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
        latent_states = h_0.repeat(1, num_examples + 1).view(batch_size, num_examples + 1, -1) # (N x L x H)
        return latent_states
