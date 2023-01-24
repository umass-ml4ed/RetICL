from typing import Optional
import torch
from torch import nn

from utils import TrainOptions
from constants import MODEL_TO_EMB_SIZE

class RetInd(nn.Module):
    def __init__(self, options: TrainOptions):
        super().__init__()
        self.options = options
        self.emb_size = MODEL_TO_EMB_SIZE[options.encoder_model]
        self.dropout = nn.Dropout(options.dropout)
        self.bilinear = nn.Parameter(torch.empty((self.emb_size, self.emb_size)))
        # Initialization follows pytorch Bilinear implementation
        init_bound = 1 / (self.emb_size ** 0.5)
        nn.init.uniform_(self.bilinear, -init_bound, init_bound)
        self.bias = nn.Parameter(torch.zeros((1)))
        self.value_fn_estimator = nn.Linear(self.emb_size, 1)

    def forward(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor,
                all_example_encodings: Optional[torch.Tensor], policy_example_indices: Optional[torch.Tensor],
                **kwargs):
        # First half of bilinear
        query_vectors = torch.matmul(current_sample_encodings, self.bilinear) # (N x E)

        # Compute activations
        batch_size, max_num_examples = example_encodings.shape[:2]
        k = all_example_encodings.shape[2]
        all_example_encodings_unrolled = all_example_encodings.view(batch_size, -1, self.emb_size) # (N x L * K x E)
        all_example_encodings_unrolled = self.dropout(all_example_encodings_unrolled)
        activations = torch.bmm(all_example_encodings_unrolled, query_vectors.unsqueeze(2)) + self.bias # (N x L * K x 1)
        activations = activations.view(-1, k) # (N * L x K)
        # Mask out previously used examples
        for used_example_idx in range(0, max_num_examples - 1):
            for next_example_idx in range(used_example_idx + 1, max_num_examples):
                activations.view(batch_size, max_num_examples, k)[
                    torch.arange(batch_size),
                    next_example_idx,
                    policy_example_indices[:, used_example_idx]
                ] = -torch.inf

        # Compute value estimates
        value_estimates = self.value_fn_estimator(query_vectors).repeat(1, max_num_examples).view(-1) # (N * L)

        return activations, value_estimates

    def get_query_vector(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor):
        return torch.matmul(current_sample_encoding, self.bilinear)
