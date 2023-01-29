import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration

class UnifiedEmbedding(torch.nn.Module):
    def __init__(self, emb_model: nn.Embedding, embed_scale) -> None:
        """Insert learned image embeddings into sentence embeddings.
        @args emb_model: nn.Embedding module from transformer pretrained models
        @args input_ids: [bs, n_shots, seq_len]
        @args attention_mask: [bs, n_shots, seq_len]
        @args img_embeddings: [bs, n_shots, hidden_size(768)]
        """
        super().__init__()
        self.embedding = emb_model
        self.embed_scale = embed_scale
    
    def insert(a, b, idx):
        """
        @args a: [bs, n_shots, seq_len, hidden_size]
        @args b: [bs, n_shots, hidden_size]
        @args idx: insert position
        """
        b = b.unsqueeze(2) # [bs, n_shots, 1, hidden_size]
        return torch.cat([a[:,:,:idx], b, a[:,:,idx:]], 2)
    
    def forward(self, input_ids, attention_mask, img_embeddings, pos):
        # TODO: assert img embedding hidden size equals to embedding layer hidden size
        # TODO: assert n_shots is equal
        # reshape input_ids into [bs*n_shots, seq_len]
        bs, n_shots = input_ids.shape[0], input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        # shape: [bs*n_shots, seq_len, hidden_size]
        input_embeds = self.embedding(input_ids) * self.embed_scale
        # reshape to [bs, n_shots, seq_len, hidden_size]
        input_embeds = input_embeds.view(bs, n_shots, input_embeds.shape[-2], -1)

        # insert into inputs at pos
        input_embeds = self.insert(input_embeds, img_embeddings, pos)

        # add attention_mask
        attention_mask = F.pad(attention_mask, pad=(1,0,0,0), mode='constant', value=1)

        # reshape to [bs, n_shots*seq_len, hidden_size]
        

        return input_embeds, attention_mask


class PromptBART(torch.nn.Module):
    def __init__(self, config) -> None:
        """BART prompt model.
        forward:
        @args input_ids: [bs, n_shots, seq_len]
        @args attention_mask: [bs, n_shots, seq_len]
        @args pos: insert postion
        @args img_embs: image embeddings obtained from image encoders
        @args labels: decoder_input_ids, 
        """
        super().__init__()
        if config.bart_model:
            self.model = BartForConditionalGeneration.from_pretrained(config.bart_model)
        else:
            self.model = BartForConditionalGeneration.from_pretrained("bart-large")

        # freeze BART
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        self.bart_model = self.model.model
        self.embedding = self.bart_model.shared # nn.Embedding
        self.embed_scale = self.bart_model.encoder.embed_scale
        self.embed_transform = UnifiedEmbedding(self.embedding, self.embed_scale)

    def forward(self, input_ids, attention_mask, img_embs, pos, labels=None):
        return
