from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import torch.nn.functional as F

from utils import device, TrainOptions, max_sbert_len

def mean_pooling(token_embeddings, attention_mask):
    # Code taken from sbert docs: https://www.sbert.net/examples/applications/computing-embeddings/README.html
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class SBERTEncoder(nn.Module):
    def __init__(self, emb_size: int, options: TrainOptions):
        super().__init__()
        self.options = options
        model_name = "sentence-transformers/" + (options.encoder_model or "all-distilroberta-v1")
        self.max_len = max_sbert_len(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO: register buffer for model
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.eos_embedding = self.model.get_input_embeddings()(torch.tensor([self.tokenizer.eos_token_id])).squeeze().to(device)
        self.input_soft_prompt = nn.Parameter(torch.zeros((options.soft_prompt_len, emb_size)))
        self.example_soft_prompt = nn.Parameter(torch.zeros((options.soft_prompt_len, emb_size)))
        nn.init.normal_(self.input_soft_prompt, mean=0.0, std=1.0)
        nn.init.normal_(self.example_soft_prompt, mean=0.0, std=1.0)

    def encode(self, seq_strings: List[str], is_example: bool):
        self.model.eval() # Don't use dropout

        # Get input tokens for given sequences
        encoded_input = self.tokenizer(
            seq_strings, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(device)
        # Extract input embeddings from model
        input_embeddings = self.model.get_input_embeddings()(encoded_input.input_ids)

        if self.options.soft_prompt_len:
            # Insert soft prompt embeddings into input sequences after start tokens
            soft_prompt = self.example_soft_prompt if is_example else self.input_soft_prompt
            input_embeddings = torch.cat([
                input_embeddings[:, 0].unsqueeze(1),
                soft_prompt.expand(input_embeddings.shape[0], -1, -1),
                input_embeddings[:, 1:]
            ], dim=1)
            # Trim to max len and make sure that sequences end with eos token
            attention_mask = F.pad(encoded_input.attention_mask, (soft_prompt.shape[0], 0), value=1)[:, :self.max_len]
            input_embeddings = input_embeddings[:, :self.max_len]
            seq_lens = attention_mask.sum(1)
            input_embeddings[torch.arange(input_embeddings.shape[0]), seq_lens - 1] = self.eos_embedding.expand(len(seq_strings), -1)
        else:
            attention_mask = encoded_input.attention_mask

        # Get output embeddings from the model
        token_embeddings = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask
        )[0]

        # Get mean-pooled outputs
        pooled_output = mean_pooling(token_embeddings, attention_mask)

        # Normalize so dot product yields fair comparison between candidates
        pooled_output = F.normalize(pooled_output, p=2, dim=1)

        return pooled_output
