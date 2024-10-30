import torch

from rs4industry.utils import get_seq_data
from rs4industry.model.module import AverageAggregator


__all__ = ["BaseQueryEncoderWithSeq"]

class BaseQueryEncoderWithSeq(torch.nn.Module):
    def __init__(self, context_embedding, item_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_encoder = item_encoder
        self.context_embedding = context_embedding
        self.seq_aggragation = AverageAggregator(dim=1)


    def forward(self, batch):
        seq_data = get_seq_data(batch)
        seq_emb = self.item_encoder(seq_data)   # BxLxD1
        seq_emb = self.seq_aggragation(seq_emb) # BxD
        context_emb = self.context_embedding(batch) # BxD2
        cat_emb = torch.cat([seq_emb, context_emb], dim=-1)
        return cat_emb
        
