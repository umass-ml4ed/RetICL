from typing import Optional
import torch
from torch import nn

from utils import TrainOptions
from constants import MODEL_TO_EMB_SIZE

class Retriever(nn.Module):
    def __init__(self, options: TrainOptions):
        super().__init__()
        self.options = options
        self.emb_size = MODEL_TO_EMB_SIZE[options.encoder_model]
        self.rnn = nn.GRU(self.emb_size, options.hidden_size, batch_first=True)
        self.h_0_transform = nn.Linear(self.emb_size, options.hidden_size)
        self.bilinear = nn.Parameter(torch.empty((options.hidden_size, self.emb_size)))
        # Initialization follows pytorch Bilinear implementation
        init_bound = 1 / (options.hidden_size ** 0.5)
        nn.init.uniform_(self.bilinear, -init_bound, init_bound)
        self.bias = nn.Parameter(torch.zeros((1)))
        self.value_fn_estimator = nn.Linear(options.hidden_size, 1)

    def forward(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor,
                top_k_example_encodings: Optional[torch.Tensor] = None, **kwargs):
        # Initial state comes from current sample
        h_0 = self.h_0_transform(current_sample_encodings)
        # Get latent state for each example in the sequence
        rnn_output, _ = self.rnn(example_encodings, h_0.unsqueeze(0))
        # For activations - use initial state and exclude last state
        latent_states = torch.cat([h_0.unsqueeze(1), rnn_output[:, :-1]], dim=1)
        # Transform from latent space to embedding space
        query_vectors = torch.matmul(latent_states, self.bilinear) # (N x L x E)
        query_vectors = query_vectors.view(-1, self.emb_size).unsqueeze(1) # (N * L x 1 x E)
        if top_k_example_encodings is None:
            # Single activation per example - complete bilinear transformation
            example_encodings = example_encodings.view(-1, self.emb_size).unsqueeze(2) # (N * L x E x 1)
            activations = torch.bmm(query_vectors, example_encodings) + self.bias # (N * L x 1 x 1)
            activations = activations.squeeze() # (N * L)
        else:
            # K activations per example - unroll and then complete bilinear transformation
            k = top_k_example_encodings.shape[2]
            top_k_unrolled = top_k_example_encodings.view(-1, k, self.emb_size).transpose(1, 2) # (N * L x E x K)
            activations = torch.bmm(query_vectors, top_k_unrolled) + self.bias # (N * L x 1 x K)
            activations = activations.squeeze() # (N * L x K)
            activations /= self.options.temp
            # TODO: mask out activations of used_idxs (prior batch policy_example_idxs)
        # Compute value estimates for baseline
        value_estimates = self.value_fn_estimator(latent_states).view(-1) # (N * L)

        return activations, value_estimates

    def get_query_vector(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor):
        h_0 = self.h_0_transform(current_sample_encoding) # Initial state comes from current sample
        if example_encodings.shape[0] == 0:
            h_t = h_0
        else:
            _, h_t = self.rnn(example_encodings, h_0.unsqueeze(0)) # Get latent state for last example in sequence
            h_t = h_t.squeeze(0)
        return torch.matmul(h_t, self.bilinear) / self.options.temp
