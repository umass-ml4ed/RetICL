from abc import abstractmethod
from typing import List
import torch
from torch import nn

from reticl.models.encoder import SBERTEncoder
from reticl.utils import TrainOptions, is_pg, orthogonal_init_
from reticl.constants import MODEL_TO_EMB_SIZE, Init, Pooling
from reticl.utils import device

class RetICLBase(nn.Module):
    def __init__(self, options: TrainOptions, use_bias: bool, mask_prev_examples: bool, num_critics: int):
        super().__init__()
        self.options = options
        self.use_bias = use_bias
        self.mask_prev_examples = mask_prev_examples
        self.emb_size = options.encoder_h or MODEL_TO_EMB_SIZE.get(options.encoder_model, 768)
        self.dropout = nn.Dropout(options.dropout)
        self.bilinear = nn.Parameter(torch.empty((options.hidden_size, self.emb_size)))
        self.bias = nn.Parameter(torch.zeros((1)))
        if options.early_stopping:
            self.early_stop_proj = nn.Linear(options.hidden_size, 1, bias=True)
            # Ensure that at initialization early stopping action has high probability relative to all examples
            # nn.init.constant_(self.early_stop_proj.bias, options.corpus_size * .02)
        self.num_critics = num_critics
        if num_critics == 0:
            self.value_fn_estimator = nn.Linear(options.hidden_size, 1)
        else:
            self.critics = nn.ParameterList([
                nn.ParameterDict({
                    "bilinear": nn.Parameter(torch.empty((options.hidden_size, self.emb_size))),
                    "bias": nn.Parameter(torch.zeros((1)))
                }) for _ in range(num_critics)
            ])
        if options.init == Init.ORTHOGONAL.value:
            nn.init.orthogonal_(self.bilinear, gain=1.0)
            if options.early_stopping:
                nn.init.orthogonal_(self.early_stop_proj.weight, gain=1.0)
            if num_critics == 0:
                orthogonal_init_(self.value_fn_estimator, gain=1.0)
            else:
                for critic in self.critics:
                    nn.init.orthogonal_(critic["bilinear"], gain=1.0)
        else:
            # Follows pytorch Bilinear implementation
            init_bound = 1 / (options.hidden_size ** 0.5)
            nn.init.uniform_(self.bilinear, -init_bound, init_bound)
        if options.ft_encoder or options.soft_prompt_len or options.pool == Pooling.ATTN.value or options.encoder_h:
            self.encoder = SBERTEncoder(options)
        else:
            self.encoder = None

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.encoder is not None and not self.options.ft_encoder:
            # If not finetuning encoder then don't save parameters to save space
            for key in list(state_dict.keys()):
                if key.startswith("encoder.model."):
                    del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.encoder is not None:
            if not any([key.startswith("encoder") for key in state_dict]):
                # If pretrained model doesn't have encoder then copy initial values
                for key, value in self.encoder.state_dict().items():
                    state_dict[f"encoder.{key}"] = value
            elif not self.options.ft_encoder:
                # If not finetuning encoder then copy initial model values
                for key, value in  self.encoder.model.state_dict().items():
                    state_dict[f"encoder.model.{key}"] = value
        return super().load_state_dict(state_dict, strict)

    @abstractmethod
    def get_latent_states(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def get_last_latent_state(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor) -> torch.Tensor:
        latent_states = self.get_latent_states(current_sample_encoding.unsqueeze(0), example_encodings.unsqueeze(0))
        return latent_states[0, -1]

    def get_activations(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor,
                        all_example_encodings: torch.Tensor, used_idxs: List[int]) -> torch.Tensor:
        if example_encodings.shape[0] == self.options.num_examples:
            # Only allow eos when max num examples used
            activations = torch.full((all_example_encodings.shape[0],), -torch.inf, device=device)
            eos_activation = torch.tensor([0], device=device)
        else:
            last_latent_state = self.get_last_latent_state(current_sample_encoding, example_encodings)
            query_vector = torch.matmul(last_latent_state, self.bilinear)
            activations = torch.matmul(query_vector, all_example_encodings.T)
            activations[used_idxs] = -torch.inf
            if example_encodings.shape[0] == 0 or not self.options.early_stopping:
                # Don't allow eos if no examples chosen or if early stopping not enabled
                eos_activation = torch.tensor([-torch.inf], device=device)
            else:
                eos_activation = self.early_stop_proj(last_latent_state)
        activations = torch.concat([activations, eos_activation])
        return activations

    def get_last_vfe(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor) -> torch.Tensor:
        vfe = self.get_vfe(current_sample_encodings, example_encodings)
        return vfe[:, -1]

    def get_vfe(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor) -> torch.Tensor:
        latent_states = self.get_latent_states(
            self.dropout(current_sample_encodings), self.dropout(example_encodings)) # (N x L x H)
        return self.value_fn_estimator(latent_states).squeeze()

    def forward(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor,
                policy_example_indices: torch.Tensor, all_example_encodings: torch.Tensor,
                seq_len: torch.Tensor, **kwargs):
        batch_size, max_seq_len = example_encodings.shape[:2]

        # Get latent states, method depends on model type
        latent_states = self.get_latent_states(
            self.dropout(current_sample_encodings), self.dropout(example_encodings)) # (N x L x H)
        latent_states = latent_states[:, :-1] # Not using representation of terminal state

        # Get query vectors (first half of bilinear)
        query_vectors = torch.matmul(latent_states, self.bilinear) # (N x L x E)

        # Compute activations
        if is_pg(self.options):
            # Compute activations over full corpus
            activations = torch.matmul(query_vectors, all_example_encodings.T) # (N x L x K)
            if self.use_bias:
                activations += self.bias
            # Mask out previously used examples
            if self.mask_prev_examples:
                for seq_idx in range(batch_size):
                    for used_example_idx in range(seq_len[seq_idx] - 1):
                        for next_example_idx in range(used_example_idx + 1, seq_len[seq_idx]):
                            activations[
                                seq_idx,
                                next_example_idx,
                                policy_example_indices[seq_idx, used_example_idx]
                            ] = -torch.inf
        else:
            # Single activation per example - complete bilinear transformation
            example_encodings = example_encodings.view(-1, self.emb_size).unsqueeze(2) # (N * L x E x 1)
            activations = torch.bmm(query_vectors.view(-1, self.emb_size).unsqueeze(1), example_encodings) + self.bias # (N * L x 1 x 1)
            activations = activations.squeeze() # (N * L)

        # Add eos action activations
        if self.options.early_stopping:
            eos_activations = self.early_stop_proj(latent_states)
            eos_activations[:, 0] = -torch.inf # Don't allow early stopping as first action
        else:
            eos_activations = torch.full((batch_size, max_seq_len, 1), -torch.inf, device=device)
        # Only allow eos when max num examples used
        if max_seq_len > self.options.num_examples:
            activations[:, -1] = -torch.inf # TODO: also consider is_pg = False
            eos_activations[:, -1] = 0

        activations = torch.concat([activations, eos_activations], dim=2)
        activations = activations.view(batch_size * max_seq_len, -1)

        # Compute value estimates
        if self.num_critics == 0:
            value_estimates = self.value_fn_estimator(latent_states).view(-1) # (N * L)
            return activations, value_estimates

        # TODO: have to handle eos for critics too, probably a less repetitive way of doing it
        critic_outputs = [
            torch.matmul(
                torch.matmul(latent_states, critic["bilinear"]).view(-1, self.emb_size),
                all_example_encodings.T
            ) + critic["bias"]
            for critic in self.critics
        ]

        return activations, critic_outputs
