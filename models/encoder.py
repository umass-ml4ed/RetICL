from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import torch.nn.functional as F

from constants import MODEL_TO_EMB_SIZE, Pooling
from utils import device, TrainOptions, max_sbert_len, orthogonal_init_

def mean_pooling(token_embeddings, attention_mask):
    # Code taken from sbert docs: https://www.sbert.net/examples/applications/computing-embeddings/README.html
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class SBERTEncoder(nn.Module):
    def __init__(self, options: TrainOptions):
        super().__init__()
        self.options = options
        emb_size = MODEL_TO_EMB_SIZE.get(options.encoder_model, 768)
        model_name = "sentence-transformers/" + (options.encoder_model or "all-distilroberta-v1")
        self.max_len = max_sbert_len(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TODO: if not ft_encoder, don't save with model params
        self.model = AutoModel.from_pretrained(model_name)
        if not options.ft_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        if options.soft_prompt_len:
            self.eos_embedding = self.model.get_input_embeddings()(torch.tensor([self.tokenizer.eos_token_id])).squeeze().to(device)
            self.input_soft_prompt = nn.Parameter(torch.zeros((options.soft_prompt_len, emb_size)))
            self.example_soft_prompt = nn.Parameter(torch.zeros((options.soft_prompt_len, emb_size)))
            nn.init.normal_(self.input_soft_prompt, mean=0.0, std=1.0)
            nn.init.normal_(self.example_soft_prompt, mean=0.0, std=1.0)
        if options.pool == Pooling.ATTN.value:
            self.attn_activator = nn.Linear(emb_size, 1, bias=False)
        if options.encoder_h:
            self.mlp = nn.Sequential(
                nn.Linear(emb_size, options.encoder_h),
                nn.ReLU()
            )
            orthogonal_init_(self.mlp)

    def encode(self, seq_strings: List[str], is_example: bool):
        if not self.options.ft_encoder:
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

        # Get pooled outputs
        if self.options.pool == Pooling.MEAN.value:
            pooled_output = mean_pooling(token_embeddings, attention_mask)
        elif self.options.pool == Pooling.ATTN.value:
            attn_activations = self.attn_activator(token_embeddings).squeeze(2)
            attn_activations[attention_mask == 0] = -torch.inf
            attn_weights = torch.softmax(attn_activations, dim=-1)
            pooled_output = torch.bmm(token_embeddings.transpose(2, 1), attn_weights.unsqueeze(2)).squeeze(2)
        else:
            raise Exception(f"{self.options.pool} pooling not supported")

        # Normalize so dot product yields fair comparison between candidates
        pooled_output = F.normalize(pooled_output, p=2, dim=1)

        # Apply MLP
        if self.options.encoder_h:
            pooled_output = self.mlp(pooled_output)

        return pooled_output
