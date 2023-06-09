import torch
from torch import nn

from models.reticl_base import RetICLBase
from utils import TrainOptions, device, orthogonal_init_
from constants import Init, ModelType

class RetICLRNN(RetICLBase):
    def __init__(self, options: TrainOptions):
        super().__init__(options)
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
        if self.lstm:
            rnn_output, _ = self.rnn(
                self.dropout(example_encodings),
                (h_0.unsqueeze(0), torch.zeros_like(h_0).unsqueeze(0).to(device))
            )
        else:
            rnn_output, _ = self.rnn(self.dropout(example_encodings), h_0.unsqueeze(0))
        latent_states = torch.cat([h_0.unsqueeze(1), rnn_output[:, :-1]], dim=1) # (N x L x H)
        return latent_states

    def get_last_latent_state(self, current_sample_encoding: torch.Tensor, example_encodings: torch.Tensor):
        h_0 = self.h_0_transform(current_sample_encoding)
        if example_encodings.shape[0] == 0:
            h_t = h_0
        else:
            # Get latent state for last example in sequence
            if self.lstm:
                _, (h_t, _) = self.rnn(
                    example_encodings,
                    (h_0.unsqueeze(0), torch.zeros_like(h_0).unsqueeze(0).to(device))
                )
            else:
                _, h_t = self.rnn(example_encodings, h_0.unsqueeze(0))
            h_t = h_t.squeeze(0)
        return h_t
