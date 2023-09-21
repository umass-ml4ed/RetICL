import torch
from torch import nn

from reticl.models.reticl_base import RetICLBase
from reticl.utils import TrainOptions, orthogonal_init_
from reticl.constants import Init

class RetICLAttn(RetICLBase):
    def __init__(self, options: TrainOptions, use_bias: bool, mask_prev_examples: bool, num_critics: int):
        super().__init__(options, use_bias, mask_prev_examples, num_critics)
        self.h_0_transform = nn.Sequential(
            nn.Linear(self.emb_size, options.hidden_size),
            nn.Tanh()
        )
        self.example_transform = nn.Sequential(
            nn.Linear(self.emb_size, options.hidden_size),
            nn.Tanh()
        )
        self.attn_activation = nn.Sequential(
            nn.Linear(options.hidden_size, 1),
        )
        if options.init == Init.ORTHOGONAL.value:
            orthogonal_init_(self.h_0_transform)
            orthogonal_init_(self.example_transform)
            orthogonal_init_(self.attn_activation)

    def _get_latent_states(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor, incl_last: bool):
        # Get sub-latent states from current sample and example encodings
        h_0 = self.h_0_transform(current_sample_encodings) # (N x H)
        if not incl_last:
            example_encodings = example_encodings[:, :-1]
        example_states = self.example_transform(example_encodings)
        sub_latent_states = torch.cat([h_0.unsqueeze(1), example_states], dim=1) # (N x L x H)
        # Get final latent states from attention weighted sums, mask so only previous states are attended to
        batch_size, num_states = sub_latent_states.shape[:2]
        attn_activations = self.attn_activation(sub_latent_states).squeeze(2) # (N x L)
        attn_activations = attn_activations.repeat(1, num_states).view(batch_size, num_states, num_states) # (N x L x L)
        tril_mask = torch.tril(torch.ones(num_states, num_states)).unsqueeze(0).repeat(batch_size, 1, 1) # (N x L x L)
        attn_activations[tril_mask == 0] = -torch.inf
        attn_weights = torch.softmax(attn_activations, dim=-1) # (N x L x L)
        latent_states = torch.bmm(attn_weights, sub_latent_states) # (N x L x H)
        return latent_states

    def get_latent_states(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor, **kwargs):
        return self._get_latent_states(current_sample_encodings, example_encodings, False)

    def get_last_latent_state(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor):
        latent_states = self._get_latent_states(current_sample_encoding.unsqueeze(0), example_encodings.unsqueeze(0), True)
        return latent_states[0, -1]
