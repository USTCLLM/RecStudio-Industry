from dataclasses import dataclass
import os
from typing import Dict, Union, Tuple

import torch
from rs4industry.model.base import BaseModel
from rs4industry.data.dataset import DataAttr4Model, ItemDataset

@dataclass
class RetrieverModelOutput:
    pos_score: torch.tensor = None
    neg_score: torch.tensor = None
    log_pos_prob: torch.tensor = None
    log_neg_prob: torch.tensor = None
    query_vector: torch.tensor = None
    pos_item_vector: torch.tensor = None
    neg_item_vector: torch.tensor = None
    pos_item_id: torch.tensor = None
    neg_item_id: torch.tensor = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    


class BaseRetriever(BaseModel):
    def __init__(
            self, 
            data_config: DataAttr4Model,
            model_config: Union[Dict, str],
            *args, **kwargs
        ):
        super().__init__(data_config, model_config, "retriever", *args, **kwargs)
        self.num_items: int = self.data_config.num_items
        self.fiid: str = self.data_config.fiid  # item id field
        self.flabel: str = self.data_config.flabels[0]  # label field, only support one label for retriever
        self.item_vectors = None
        self.item_ids = None

    def init_modules(self):
        # super().init_modules()
        self.item_encoder = self.get_item_encoder()
        self.query_encoder = self.get_query_encoder()
        self.score_function = self.get_score_function()
        self.negative_sampler = self.get_negative_sampler()
        self.loss_function = self.get_loss_function()

    
    def get_query_encoder(self):
        raise NotImplementedError

    def get_item_encoder(self):
        raise NotImplementedError

    def get_score_function(self):
        raise NotImplementedError

    def get_negative_sampler(self):
        return None

    def get_loss_function(self):
        raise NotImplementedError

    
    def score(
            self, 
            batch,
            inference=False,
            item_loader=None,
            *args, 
            **kwargs
        ):
        pos_item_id = batch[self.fiid]
        query_vec = self.query_encoder(batch)
        pos_item_vec = self.item_encoder(batch)
        pos_score = self.score_function(query_vec, pos_item_vec)

        if not inference:
            # training mode, sampling negative items or use all items as negative items
            if self.negative_sampler:
                # sampling negative items
                if not self.model_config.num_neg:
                    raise ValueError("`negative_count` is required when `sampler` is not none.")
                else:
                    neg_item_idx, log_neg_prob = self.sampling(query_vec, self.model_config.num_neg)
                    neg_item_feat = self.get_item_feat(item_loader.dataset, neg_item_idx)
                    neg_item_id = neg_item_feat.get(self.fiid)
                    neg_item_vec = self.item_encoder(neg_item_feat)
                    # log_pos_prob = self.negative_sampler.compute_item_p(query_vec, pos_item_id)
                    log_pos_prob = None
            else:
                # no negative sampling, use all items as negative items, such as full softmax
                neg_item_id = None
                neg_item_vec, _ = self.get_item_vectors(item_loader)
                neg_item_vec = self.item_vectors
                log_pos_prob, log_neg_prob = None, None
            neg_score = self.score_function(query_vec, neg_item_vec)
        else:
            neg_score = None
            neg_item_id = None
            neg_item_vec = None
            log_pos_prob, log_neg_prob = None, None

        output = RetrieverModelOutput(
            pos_score=pos_score,
            neg_score=neg_score,
            log_pos_prob=log_pos_prob,
            log_neg_prob=log_neg_prob,
            query_vector=query_vec,
            pos_item_vector=pos_item_vec,
            neg_item_vector=neg_item_vec,
            pos_item_id=pos_item_id,
            neg_item_id=neg_item_id
        )
        return output
    

    def sampling(self, query, num_neg, *args, **kwargs):
        return self.negative_sampler(query, num_neg)
        
    
    def forward(self, batch, *args, **kwargs) -> RetrieverModelOutput:
        output = self.score(batch, *args, **kwargs)
        return output

    def cal_loss(self, batch, *args, **kwargs) -> Dict:
        output = self.forward(batch, *args, **kwargs)
        output_dict = output.to_dict()
        labels = batch[self.flabel]
        output_dict['label'] = labels
        loss = self.loss_function(**output_dict)
        if isinstance(loss, dict):
            return loss
        else:
            return {'loss': loss}
        
    @torch.no_grad()
    def eval_step(self, batch, k, user_hist=None, chunk_size=2048, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        query_vec = self.query_encoder(batch)
        pos_vec = self.item_encoder(batch)
        pos_scores = self.score_function(query_vec, pos_vec)
        scores = self.score_all_items(query_vec, chunk_size=chunk_size)
        more = user_hist.size(1) if user_hist is not None else 0
        score, topk_idx = torch.topk(scores, k + more, dim=-1)
        # we usually do not mask history in industrial settings
        if pos_scores.dim() < score.dim():
            pos_scores = pos_scores.unsqueeze(-1)
        all_scores = torch.cat([pos_scores, score], dim=-1) # [B, N + 1]
        # sort and get the index of the first item
        _, indice = torch.sort(all_scores, dim=-1, descending=True, stable=True)
        pred = indice[:, :k] == 0
        target = torch.ones_like(batch[self.fiid], dtype=torch.bool).view(-1, 1)   # [B, 1]
        return pred, target


    @torch.no_grad()
    def predict(self, context_input: dict) -> torch.Tensor:
        """ Encode context input, output vectors.
        Args:
            context_input (dict): context input features, e.g., user_id, session_id, etc.
        Returns:
            torch.Tensor: [B, D], where B is batch size and D is embedding dimension.
        """
        context_vec = self.query_encoder(context_input)
        return context_vec

    def get_item_feat(self, item_dataset: ItemDataset, item_id: torch.tensor):
        """
        Get item features by item id.
        Args:
            item_id: [B]
        Returns:
            item_feat: [B, N]
        """
        item_feat = item_dataset.get_item_feat(item_id)
        return item_feat

    def get_item_vectors(self, item_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        all_item_vec = []
        all_item_id = []
        device = next(self.parameters()).device
        for item_batch in item_loader:
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            item_vec = self.item_encoder(item_batch)
            all_item_vec.append(item_vec)
            all_item_id.append(item_batch[self.fiid].cpu())
        all_item_vec = torch.cat(all_item_vec, dim=0)   # [N, D]
        all_item_id = torch.cat(all_item_id, dim=0) # [N]
        return all_item_vec, all_item_id

    @torch.no_grad()
    def update_item_vectors(self, item_loader) -> None:
        """ Update item vectors and item ids.

        The method does not record the gradients. If you want to retain the gradients in getting item vectors,
        use `get_item_vectors` instead.

        Args:
            item_loader (DataLoader): item loader.
        """
        # -> [N, D]
        with torch.no_grad():
            print("Updating item vectors...")
            self.item_vectors, self.item_ids = self.get_item_vectors(item_loader)
            print("Done")


    def score_all_items(self, query_vec: torch.Tensor, chunk_size=2048):
        """ Score all items once, if OOM, then split the item vectors into chunks
        Args:
            query_vec (torch.Tensor): [B, D]
            chunk_size (int): chunk size of item vectors. default is 2048.
        
        Returns:
            scores (torch.Tensor): [B, N]
        """
        try:
            # try to score all items once, if OOM, then split the item vectors into chunks
            scores = self.score_function(query_vec, self.item_vectors)
        except torch.cuda.OutOfMemoryError:
            item_vectors = self.item_vectors
            scores = []
            for i in range(0, len(item_vectors), chunk_size):
                chunk = item_vectors[i:i+chunk_size]
                scores.append(self.score_function(query_vec, chunk))
            scores = torch.cat(scores, dim=0)
        return scores


    def save_item_vectors(self, checkpoint_dir: str, item_loader=None):
        if self.item_vectors is None or self.item_ids is None:
            self.update_item_vectors(item_loader=item_loader)
        checkpoint_path = os.path.join(checkpoint_dir, 'item_vectors.pt')
        torch.save({'item_vectors': self.item_vectors, 'item_ids': self.item_ids}, checkpoint_path)
