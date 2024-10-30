from typing import OrderedDict
import torch
from rs4industry.config import ModelArguments
from rs4industry.model.retriever import BaseRetriever
from rs4industry.model.utils import get_modules
from rs4industry.model.module import MultiFeatEmbedding, MLPModule


__all__ = ["MLPRetriever", "MLPRetrieverConfig"]

class MLPRetrieverConfig(ModelArguments):
    embedding_dim: int = 10
    mlp_layers: list[int] = [128, 128]
    activation: str = "relu"
    dropout: float = 0.3
    batch_norm: bool = True
    

class MLPRetriever(BaseRetriever):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def get_item_encoder(self):
        item_emb = MultiFeatEmbedding(
            features=self.data_config.item_features,
            stats=self.data_config.stats,
            embedding_dim=self.model_config.embedding_dim,
            concat_embeddings=True
        )
        mlp = MLPModule(
            mlp_layers= [item_emb.total_embedding_dim] + self.model_config.mlp_layers,
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        return torch.nn.Sequential(OrderedDict([
            ("item_embedding", item_emb),
            ("mlp", mlp)
            ]))
    

    def get_query_encoder(self):
        context_emb = MultiFeatEmbedding(
            features=self.data_config.context_features,
            stats=self.data_config.stats,
            embedding_dim=self.model_config.embedding_dim
        )
        base_encoder = get_modules("encoder", "BaseQueryEncoderWithSeq")(
            context_embedding=context_emb,
            item_encoder=self.item_encoder
        )
        output_dim = self.model_config.mlp_layers[-1] + context_emb.total_embedding_dim
        mlp = MLPModule(
            mlp_layers= [output_dim] + self.model_config.mlp_layers,
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )

        return torch.nn.Sequential(OrderedDict([
            ("encoder", base_encoder),
            ("mlp", mlp)
            ]))
    

    def get_score_function(self):
        return get_modules("score", "InnerProductScorer")()
    
    def get_loss_function(self):
        return get_modules("loss", "BPRLoss")()

    
    def get_negative_sampler(self):
        sampler_cls = get_modules("sampler", "UniformSampler")
        return sampler_cls(num_items=self.data_config.num_items)
        
