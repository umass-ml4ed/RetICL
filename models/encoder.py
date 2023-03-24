from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from models.retriever import Retriever
from utils import device

def mean_pooling(token_embeddings, attention_mask):
    # Code taken from sbert docs: https://www.sbert.net/examples/applications/computing-embeddings/README.html
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class SBERTEncoder:
    def __init__(self, encoder_model: str, retriever: Retriever):
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model or "all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained(encoder_model or "all-mpnet-base-v2")
        self.retriever = retriever
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, seq_strings: List[str], is_example: bool):
        encoded_input = self.tokenizer(
            seq_strings, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        input_embeddings = self.model.get_input_embeddings()(encoded_input.input_ids)
        soft_prompt = self.retriever.example_soft_prompt if is_example else self.retriever.input_soft_prompt
        # TODO: check that we need to do thing with CLS token and that this is right...
        input_embeddings = torch.cat([
            input_embeddings[:, 0].unsqueeze(1),
            soft_prompt.expand(input_embeddings.shape[0], -1, -1),
            input_embeddings[:, 1:]
        ])
        attention_mask = F.pad(encoded_input.attention_mask, (soft_prompt.shape[0], 0), value=1)

        token_embeddings = self.model(
            input_embeds=input_embeddings,
            attention_mask=attention_mask
        )[0]

        return mean_pooling(token_embeddings, attention_mask)
