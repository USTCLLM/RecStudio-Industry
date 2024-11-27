from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
from rs4industry.model.base import BaseModel
from rs4industry.model.utils import get_modules
from rs4industry.data.dataset import DataAttr4Model, ItemDataset
from rs4industry.model.module import MultiFeatEmbedding
from rs4industry.utils import split_batch


@dataclass
class RankerModelOutput:
    score: torch.Tensor
    embedding: torch.Tensor

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class BaseRanker(BaseModel):
    def __init__(
            self, 
            data_config: DataAttr4Model,
            model_config: Union[Dict, str],
            *args, **kwargs
        ):
        self.num_seq_feat = len(data_config.seq_features)
        self.num_context_feat = len(data_config.context_features)
        self.num_item_feat = len(data_config.item_features)
        self.num_feat = self.num_seq_feat + self.num_context_feat + self.num_item_feat
        super().__init__(data_config, model_config, "retriever", *args, **kwargs)
        self.model_type = "ranker"
        self.num_items: int = self.data_config.num_items
        self.fiid: str = self.data_config.fiid  # item id field
        # label fields, base ranker only support one label
        # if need multiple labels, use MultiTaskRanker instead
        self.flabel: str = self.data_config.flabels[0]

    
    def init_modules(self):
        self.embedding_layer = self.get_embedding_layer()
        self.sequence_encoder = self.get_sequence_encoder()
        self.feature_interaction_layer = self.get_feature_interaction_layer()
        self.prediction_layer = self.get_prediction_layer()
        self.loss_function = self.get_loss_function()

    def get_embedding_layer(self):
        emb = MultiFeatEmbedding(
            features=self.data_config.features,
            stats=self.data_config.stats,
            embedding_dim=self.model_config.embedding_dim,
            concat_embeddings=False,
            stack_embeddings=True
        )
        return emb
    
    def get_sequence_encoder(self):
        raise NotImplementedError


    def get_feature_interaction_layer(self):
        raise NotImplementedError

    
    def get_predition_layer(self):
        raise NotImplementedError

    def get_loss_function(self):
        # BCELoss is not good for autocast in distributed training, reminded by pytorch
        return get_modules("loss", "BCEWithLogitLoss")(reduction='mean')
    

    def forward(self, batch, cal_loss=False, *args, **kwargs) -> RankerModelOutput:
        if cal_loss:
            return self.cal_loss(batch, *args, **kwargs)
        else:
            output = self.score(batch, *args, **kwargs)
        return output
    

    def cal_loss(self, batch, *args, **kwargs) -> Dict:
        label = batch[self.flabel].float()
        output = self.forward(batch, *args, **kwargs)
        output_dict = output.to_dict()
        output_dict['label'] = label
        loss = self.loss_function(**output_dict)
        if isinstance(loss, dict):
            return loss
        else:
            return {'loss': loss}
    

    def score(self, batch, *args, **kwargs) -> RankerModelOutput:
        context_feat, seq_feat, item_feat = split_batch(batch, self.data_config)
        all_embs = []
        if len(seq_feat) > 0:
            seq_emb = self.embedding_layer(seq_feat, strict=False)
            seq_rep = self.sequence_encoder(seq_emb)   # [B, N1, D]
            all_embs.append(seq_rep)
        context_emb = self.embedding_layer(context_feat, strict=False)  # [B, N2, D]
        item_emb = self.embedding_layer(item_feat, strict=False)    # [B, N3, D]
        all_embs += [context_emb, item_emb]
        all_embs = torch.concat(all_embs, dim=1) # [B, N1+N2+N3, D]
        interacted_emb = self.feature_interaction_layer(all_embs)    # [B, **]
        score = self.prediction_layer(interacted_emb)   # [B], sigmoid
        if len(score.shape) == 2 and score.size(-1) == 1:
            score = score.squeeze(-1)   # [B, 1] -> [B]
        return RankerModelOutput(score, [context_emb, item_emb, seq_emb])
    

    @torch.no_grad()
    def predict(self, context_input: Dict, candidates: Dict, topk: int, gpu_mem_save=False, *args, **kwargs):
        """ predict topk candidates for each context
        
        Args:
            context_input (Dict): input context feature
            candidates (Dict): candidate items
            topk (int): topk candidates
            gpu_mem_save (bool): whether to save gpu memroy by using loop to process each candidate

        Returns:
            torch.Tensor: topk indices (offset instead of real item id)
        """
        num_candidates = candidates[self.fiid].size(1)
        if not gpu_mem_save:
            # expand batch to match the number of candidates, consuming more memory
            batch_size = candidates[self.fiid].size(0)
            for k, v in context_input.items():
                # B, * -> BxN, *
                if isinstance(v, dict):
                    for k_, v_ in v.items():
                        v[k_] = v_.repeat_interleave(num_candidates, dim=0)
                else:
                    context_input[k] = v.repeat_interleave(num_candidates, dim=0)
            for k, v in candidates.items():
                # B, N, * -> BxN, *
                candidates[k] = v.view(-1, *v.shape[2:])
            context_input.update(candidates)    # {key: BxN, *}
            output = self.score(context_input, *args, **kwargs)
            scores = output.score.view(batch_size, num_candidates)  # [B, N]
        else:
            # use loop to process each candidate
            scores = []
            for i in range(num_candidates):
                candidate = {k: v[:, i] for k, v in candidates.items()}
                new_batch = dict(**context_input)
                new_batch.update(candidate)
                output = self.score(new_batch, *args, **kwargs)
                scores.append(output.score)
            scores = torch.stack(scores, dim=-1)    # [B, N]
        
        # get topk idx
        topk_score, topk_idx = torch.topk(scores, topk)
        return topk_idx
    
    

    @torch.no_grad()
    def eval_step(self, batch, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.score(batch, *args, **kwargs)
        score = output.score
        # check if the last layer of the model is a sigmoid function
        # get the last layer of the model
        pred = score
        target = batch[self.flabel].float()
        return pred, target

    
