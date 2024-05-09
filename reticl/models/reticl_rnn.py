import torch
from torch import nn

from reticl.models.reticl_base import RetICLBase
from reticl.utils import TrainOptions, device, orthogonal_init_
from reticl.constants import Init, ModelType

class RetICLRNN(RetICLBase):
    def __init__(self, options: TrainOptions, use_bias: bool, mask_prev_examples: bool, num_critics: int):
        super().__init__(options, use_bias, mask_prev_examples, num_critics)
        self.lstm = options.model_type.startswith(ModelType.LSTM.value)
        self.concat = options.model_type == ModelType.LSTM_CONCAT.value
        self.first = options.model_type == ModelType.LSTM_FIRST.value
        if self.lstm:
            self.rnn = nn.LSTM(self.emb_size, options.hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(self.emb_size, options.hidden_size, batch_first=True)
        if self.concat:
            self.hidden_transform = nn.Sequential(
                nn.Linear(self.emb_size + options.hidden_size, options.hidden_size),
                nn.Dropout(options.dropout),
                nn.Tanh(),
                nn.Linear(options.hidden_size, options.hidden_size),
                nn.Tanh()
            )
        elif not self.first:
            self.h_0_transform = nn.Sequential(
                nn.Linear(self.emb_size, options.hidden_size),
                nn.Dropout(options.dropout),
                nn.Tanh()
            )
        if options.init == Init.ORTHOGONAL.value:
            orthogonal_init_(self.rnn)
            if self.concat:
                orthogonal_init_(self.hidden_transform)
            elif not self.first:
                orthogonal_init_(self.h_0_transform)

    def get_latent_states(self, current_sample_encodings: torch.Tensor, example_encodings: torch.Tensor, **kwargs):
        if self.first:
            inputs = torch.concat([current_sample_encodings.unsqueeze(1), example_encodings], dim=1)
            return self.rnn(inputs)[0]
        if self.concat:
            h_0 = torch.zeros((current_sample_encodings.shape[0], self.options.hidden_size)).to(device)
        else:
            h_0 = self.h_0_transform(current_sample_encodings) # (N x H)
            if example_encodings.shape[1] == 0:
                return h_0.unsqueeze(1)
        if self.concat and example_encodings.shape[1] == 0:
            rnn_output = torch.zeros((current_sample_encodings.shape[0], 0, self.options.hidden_size)).to(device)
        else:
            if self.lstm:
                rnn_output, _ = self.rnn(
                    example_encodings,
                    (h_0.unsqueeze(0), torch.zeros_like(h_0).unsqueeze(0).to(device))
                )
            else:
                rnn_output, _ = self.rnn(example_encodings, h_0.unsqueeze(0))
        if self.concat:
            rnn_output = torch.concat([
                torch.zeros((rnn_output.shape[0], 1, self.options.hidden_size)).to(device), rnn_output], dim=1)
            sub_latent_states = torch.concat([
                current_sample_encodings.unsqueeze(1).repeat(1, rnn_output.shape[1], 1),
                rnn_output
            ], dim=2)
            latent_states = self.hidden_transform(sub_latent_states)
        else:
            latent_states = torch.cat([h_0.unsqueeze(1), rnn_output], dim=1) # (N x L x H)
        return latent_states
