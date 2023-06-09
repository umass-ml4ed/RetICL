from abc import abstractmethod
import torch
from torch import nn

from models.encoder import SBERTEncoder
from utils import TrainOptions, is_pg, orthogonal_init_
from constants import MODEL_TO_EMB_SIZE, Init, Pooling

class RetICLBase(nn.Module):
    def __init__(self, options: TrainOptions):
        super().__init__()
        self.options = options
        self.emb_size = options.encoder_h or MODEL_TO_EMB_SIZE.get(options.encoder_model, 768)
        self.dropout = nn.Dropout(options.dropout)
        self.bilinear = nn.Parameter(torch.empty((options.hidden_size, self.emb_size)))
        self.bias = nn.Parameter(torch.zeros((1)))
        self.value_fn_estimator = nn.Linear(options.hidden_size, 1)
        if options.init == Init.ORTHOGONAL.value:
            nn.init.orthogonal_(self.bilinear, gain=1.0)
            orthogonal_init_(self.value_fn_estimator, gain=1.0)
        else:
            # Follows pytorch Bilinear implementation
            init_bound = 1 / (options.hidden_size ** 0.5)
            nn.init.uniform_(self.bilinear, -init_bound, init_bound)
        if options.ft_encoder or options.soft_prompt_len or options.pool == Pooling.ATTN.value or options.encoder_h:
            self.encoder = SBERTEncoder(options)
        else:
            self.encoder = None

    @abstractmethod
    def get_latent_states(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_last_latent_state(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_query_vector(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.get_last_latent_state(current_sample_encoding, example_encodings), self.bilinear)

    def get_last_vfe(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor) -> torch.Tensor:
        h_t = self.get_last_latent_state(current_sample_encoding, example_encodings)
        return self.value_fn_estimator(h_t).squeeze()

    def forward(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor,
                policy_example_indices: torch.Tensor, all_example_encodings: torch.Tensor,
                **kwargs):
        # Get latent states, method depends on model type
        latent_states = self.get_latent_states(current_sample_encodings, self.dropout(example_encodings)) # (N x L x H)

        # Get query vectors (first half of bilinear)
        query_vectors = torch.matmul(latent_states, self.bilinear).view(-1, self.emb_size) # (N * L x E)

        # Compute activations
        if is_pg(self.options):
            # Compute activations over full corpus
            batch_size, max_num_examples = example_encodings.shape[:2]
            activations = torch.matmul(query_vectors, all_example_encodings.T) # (N * L x K)
            # Mask out previously used examples
            for used_example_idx in range(0, max_num_examples - 1):
                for next_example_idx in range(used_example_idx + 1, max_num_examples):
                    activations.view(batch_size, max_num_examples, -1)[
                        torch.arange(batch_size),
                        next_example_idx,
                        policy_example_indices[:, used_example_idx]
                    ] = -torch.inf
        else:
            # Single activation per example - complete bilinear transformation
            example_encodings = example_encodings.view(-1, self.emb_size).unsqueeze(2) # (N * L x E x 1)
            activations = torch.bmm(query_vectors.unsqueeze(1), example_encodings) + self.bias # (N * L x 1 x 1)
            activations = activations.squeeze() # (N * L)

        # Compute value estimates
        value_estimates = self.value_fn_estimator(latent_states).view(-1) # (N * L)

        return activations, value_estimates
