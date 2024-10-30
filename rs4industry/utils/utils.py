from operator import itemgetter
from typing import Dict, Tuple

import torch
from rs4industry.data.dataset import DataAttr4Model


__all__ =[
    "get_seq_data",
    "split_batch",
    "batch_to_device"
]

def get_seq_data(d: dict):
    if "seq" in d:
        return d['seq']
    else:
        return {}


def split_batch(batch: dict, data_attr: DataAttr4Model) -> Tuple[Dict, Dict, Dict]:
    context_feat = {}; item_feat = {}
    seq_feat = get_seq_data(batch)
    for k, v in batch.items():
        if k in data_attr.context_features:
            context_feat[k] = v
        elif k in data_attr.item_features:
            item_feat[k] = v
    return context_feat, seq_feat, item_feat


def batch_to_device(batch, device) -> Dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = batch_to_device(value, device)
    return batch
