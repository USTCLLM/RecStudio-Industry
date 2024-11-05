import json
import os
from typing import Dict, Union
import torch
import torch.nn as nn

from rs4industry.data.dataset import DataAttr4Model
from rs4industry.config.model import ModelArguments
from rs4industry.model.utils import get_model_cls


class BaseModel(nn.Module):
    def __init__(
            self,
            data_config: DataAttr4Model,
            model_config: Union[Dict, str],
            model_type: str,
            *args, **kwargs
        ):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.data_config: DataAttr4Model = data_config
        self.model_config: ModelArguments = self.load_config(model_config)
        self.model_type = model_type
        # self.loggers = get_logger(training_config)
        self.init_modules()

    def load_config(self, config: Union[Dict, str]) -> ModelArguments:
        if isinstance(config, ModelArguments):
            config = config
        elif isinstance(config, str):
            with open(config, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config = ModelArguments.from_dict(config_dict)
        elif isinstance(config, dict):
            config = ModelArguments.from_dict(config)
        else:
            raise ValueError(f"Config should be either a dictionary or a path to a JSON file, got {type(config)} instead.")
        return config


    @staticmethod
    def from_pretrained(checkpoint_dir: str):
        config_path = os.path.join(checkpoint_dir, "model_config.json")
        with open(config_path, "r", encoding="utf-8") as config_path:
            config_dict = json.load(config_path)
        data_attr = DataAttr4Model.from_dict(config_dict['data_attr'])
        model_cls = get_model_cls(config_dict['model_type'], config_dict['model_name'])
        del config_dict['data_attr'], config_dict['model_type'], config_dict['model_name']
        model_config = ModelArguments.from_dict(config_dict)
        ckpt_path = os.path.join(checkpoint_dir, "model.pt")
        state_dict = torch.load(ckpt_path, weights_only=True)
        model = model_cls(data_attr, model_config)
        if "item_vectors" in state_dict:
            model.item_vectors = state_dict["item_vectors"]
            del state_dict['item_vectors']
        model.load_state_dict(state_dict=state_dict, strict=True)
        return model


    def save(self, checkpoint_dir: str, **kwargs):
        self.save_checkpoint(checkpoint_dir)
        self.save_configurations(checkpoint_dir)


    def score(self, batch, *args, **kwargs):
        return NotImplementedError

    def forward(self, batch, *args, **kwargs):
        raise NotImplementedError

    def cal_loss(self, batch, *args, **kwargs):
        raise NotImplementedError
    

    def validation(self, test_loader):
        raise NotImplementedError
    

    def prediction(self, test_loader):
        raise NotImplementedError
    

    def save_checkpoint(self, checkpoint_dir: str):
        path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(self.state_dict(), path)
        
    

    def save_configurations(self, checkpoint_dir: str):
        path = os.path.join(checkpoint_dir, "model_config.json")
        config_dict = self.model_config.to_dict()
        config_dict['model_type'] = self.model_type
        config_dict['model_name'] = self.__class__.__name__
        config_dict['data_attr'] = self.data_config.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
    

    def init_modules(self):
        raise NotImplementedError
