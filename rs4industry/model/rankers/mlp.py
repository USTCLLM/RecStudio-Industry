import torch
from rs4industry.model.ranker import BaseRanker
from rs4industry.model.utils import get_modules
from rs4industry.model.module import MLPModule, LambdaModule



class MLPRanker(BaseRanker):
    def get_sequence_encoder(self):
        cls = get_modules("module", "AverageAggregator")
        encoder = cls(dim=1)
        return encoder
        
    
    def get_feature_interaction_layer(self):
        flatten_layer = LambdaModule(lambda x: x.flatten(start_dim=1))  # [B, N, D] -> [B, N*D]
        mlp_layer = MLPModule(
            mlp_layers= [self.num_feat * self.model_config.embedding_dim] + self.model_config.mlp_layers,
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        return torch.nn.Sequential(flatten_layer, mlp_layer)
    

    def get_prediction_layer(self):
        pred_mlp = MLPModule(
            mlp_layers=[self.model_config.mlp_layers[-1]] + self.model_config.prediction_layers + [1],
            activation_func=self.model_config.activation,
            dropout=self.model_config.dropout,
            bias=True,
            batch_norm=self.model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        sigmoid = torch.nn.Sigmoid()
        return torch.nn.Sequential(pred_mlp, sigmoid)
    