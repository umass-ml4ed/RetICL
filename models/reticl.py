from typing import Optional
import torch
from torch import nn

from utils import TrainOptions
from constants import MODEL_TO_EMB_SIZE

class RetICL(nn.Module):
    def __init__(self, options: TrainOptions):
        super().__init__()
        self.options = options
        self.emb_size = MODEL_TO_EMB_SIZE[options.encoder_model]
        # self.rnn = nn.GRU(self.emb_size, options.hidden_size, batch_first=True)
        self.rnn = nn.RNN(self.emb_size, options.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(options.dropout)
        self.h_0_transform = nn.Sequential(
            nn.Linear(self.emb_size, options.hidden_size),
            nn.Tanh()
        )
        self.bilinear = nn.Parameter(torch.empty((options.hidden_size, self.emb_size)))
        # Initialization follows pytorch Bilinear implementation
        init_bound = 1 / (options.hidden_size ** 0.5)
        nn.init.uniform_(self.bilinear, -init_bound, init_bound)
        self.bias = nn.Parameter(torch.zeros((1)))
        self.value_fn_estimator = nn.Linear(options.hidden_size, 1)

    def get_latent_states_and_query_vectors(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor, **kwargs):
        # Initial state comes from current sample
        h_0 = self.h_0_transform(current_sample_encodings)
        # Get latent state for each example in the sequence
        rnn_output, _ = self.rnn(self.dropout(example_encodings), h_0.unsqueeze(0))
        # For activations - use initial state and exclude last state
        latent_states = torch.cat([h_0.unsqueeze(1), rnn_output[:, :-1]], dim=1)
        # Transform from latent space to embedding space
        query_vectors = torch.matmul(latent_states, self.bilinear) # (N x L x E)
        return latent_states, query_vectors

    def forward(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor,
                all_example_encodings: Optional[torch.Tensor] = None, policy_example_indices: Optional[torch.Tensor] = None,
                **kwargs):
        # Run RNN and first half of bilinear
        latent_states, query_vectors = self.get_latent_states_and_query_vectors(current_sample_encodings, example_encodings)
        query_vectors = query_vectors.view(-1, self.emb_size).unsqueeze(1) # (N * L x 1 x E)

        # Compute activations
        if all_example_encodings is None:
            # Single activation per example - complete bilinear transformation
            example_encodings = example_encodings.view(-1, self.emb_size).unsqueeze(2) # (N * L x E x 1)
            activations = torch.bmm(query_vectors, example_encodings) + self.bias # (N * L x 1 x 1)
            activations = activations.squeeze() # (N * L)
        else:
            # K activations per example - unroll and then complete bilinear transformation
            batch_size, max_num_examples = example_encodings.shape[:2]
            k = all_example_encodings.shape[2]
            all_example_encodings_unrolled = all_example_encodings.view(-1, k, self.emb_size).transpose(1, 2) # (N * L x E x K)
            all_example_encodings_unrolled = self.dropout(all_example_encodings_unrolled)
            activations = torch.bmm(query_vectors, all_example_encodings_unrolled) + self.bias # (N * L x 1 x K)
            activations = activations.squeeze() # (N * L x K)
            # Mask out previously used examples
            for used_example_idx in range(0, max_num_examples - 1):
                for next_example_idx in range(used_example_idx + 1, max_num_examples):
                    activations.view(batch_size, max_num_examples, k)[
                        torch.arange(batch_size),
                        next_example_idx,
                        policy_example_indices[:, used_example_idx]
                    ] = -torch.inf

        # Compute value estimates
        value_estimates = self.value_fn_estimator(latent_states).view(-1) # (N * L)

        return activations, value_estimates

    def get_query_vector(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor):
        h_0 = self.h_0_transform(current_sample_encoding) # Initial state comes from current sample
        if example_encodings.shape[0] == 0:
            h_t = h_0
        else:
            _, h_t = self.rnn(example_encodings, h_0.unsqueeze(0)) # Get latent state for last example in sequence
            h_t = h_t.squeeze(0)
        return torch.matmul(h_t, self.bilinear)

    def get_all_value_estimates(self, current_sample_encoding: torch.Tensor,
                                all_example_encodings: torch.Tensor = None, **kwargs):
        h_0 = self.h_0_transform(current_sample_encoding).repeat(all_example_encodings.shape[0], 1) # (K x H)
        _, h_1 = self.rnn(all_example_encodings.unsqueeze(1), h_0.unsqueeze(0)) # (1 x K x H)
        return self.value_fn_estimator(h_1).view(-1) # (K)
