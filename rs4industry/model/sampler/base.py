

from typing import Optional, Union
import numpy as np
import torch
from torch import Tensor


class Sampler(object):
    def __init__(self, num_items, scorer_fn=None):
        super(Sampler, self).__init__()
        self.num_items = num_items
        self.scorer = scorer_fn

    def update(self, item_embs, max_iter=30):
        pass

    def compute_item_p(self, query, pos_items):
        pass

    def __call__(self, *args, **kwargs):
        pass


class UniformSampler(Sampler):
    """
    For each user, sample negative items
    """

    def __call__(
            self,
            query: Union[Tensor, int],
            num_neg: int,
            pos_items: Optional[Tensor] = None,
            device: Optional[torch.device] = None
        ):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L || assume padding=0
        # not deal with reject sampling
        if isinstance(query, int):
            num_queries = query
            device = pos_items.device if pos_items is not None else device
            shape = (num_queries, )
        elif isinstance(query, Tensor):
            query = query
            num_queries = np.prod(query.shape[:-1])
            device = query.device
            shape = query.shape[:-1]

        with torch.no_grad():
            neg_items = torch.randint(0, self.num_items, size=(num_queries, num_neg), device=device)  # padding with zero
            neg_items = neg_items.reshape(*shape, -1)  # B x L x Neg || B x Neg
            neg_prob = self.compute_item_p(None, neg_items)
            if pos_items is not None:
                pos_prob = self.compute_item_p(None, pos_items)
                return pos_prob, neg_items, neg_prob
            else:
                return neg_items, neg_prob

    def compute_item_p(self, query, pos_items):
        return torch.zeros_like(pos_items)