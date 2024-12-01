from rs4industry.trainer import Trainer
from rs4industry.model.base import BaseModel
from rs4industry.data.dataset import get_datasets

data_config_path = "../config/data/recflow_ranker.json"
train_config_path = "../config/mlp_ranker/train.json"

(train_data, eval_data), data_config = get_datasets(data_config_path)

path_to_eval_ckpt="xxxx"

model = BaseModel.from_pretrained(path_to_eval_ckpt)

print(model)

trainer = Trainer(model, train=False)

trainer.config.metrics = ["auc", "logloss"]
trainer.config.eval_batch_size = 2048

trainer.evaluation(eval_data)
