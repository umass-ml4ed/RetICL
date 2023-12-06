import torch
from torch import nn

from reticl.models.reticl_base import RetICLBase
from reticl.utils import TrainOptions, device, orthogonal_init_
from reticl.constants import Init, ModelType

class RetICLRNN(RetICLBase):
    def __init__(self, options: TrainOptions, use_bias: bool, mask_prev_examples: bool, num_critics: int):
        super().__init__(options, use_bias, mask_prev_examples, num_critics)
        self.lstm = options.model_type == ModelType.LSTM.value
        if self.lstm:
            self.rnn = nn.LSTM(self.emb_size, options.hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(self.emb_size, options.hidden_size, batch_first=True)
        self.h_0_transform = nn.Sequential(
            nn.Linear(self.emb_size, options.hidden_size),
            nn.Tanh()
        )
        if options.init == Init.ORTHOGONAL.value:
            orthogonal_init_(self.rnn)
            orthogonal_init_(self.h_0_transform)

    def get_latent_states(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor, **kwargs):
        h_0 = self.h_0_transform(current_sample_encodings) # (N x H)
        if example_encodings.shape[1] == 0:
            return h_0.unsqueeze(1)
        if self.lstm:
            rnn_output, _ = self.rnn(
                example_encodings,
                (h_0.unsqueeze(0), torch.zeros_like(h_0).unsqueeze(0).to(device))
            )
        else:
            rnn_output, _ = self.rnn(example_encodings, h_0.unsqueeze(0))
        latent_states = torch.cat([h_0.unsqueeze(1), rnn_output], dim=1) # (N x L x H)
        return latent_states
