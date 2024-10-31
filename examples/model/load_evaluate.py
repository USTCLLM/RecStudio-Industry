from rs4industry.model.base import BaseModel
from rs4industry.data.dataset import get_datasets
from rs4industry.trainer import Trainer

data_config_path = "/data1/home/recstudio/huangxu/rec-studio-industry/examples/config/data/recflow_ranker.json"
train_config_path = "/data1/home/recstudio/huangxu/rec-studio-industry/examples/config/mlp_ranker/train.json"

(train_data, eval_data), data_config = get_datasets(data_config_path)
model = BaseModel.from_pretrained("/data1/home/recstudio/huangxu/saves/mlp_ranker_test2/best_ckpt")
print(model)

trainer = Trainer(model, train=False)

trainer.config.metrics = ["auc", "logloss"]
trainer.config.eval_batch_size = 2048

trainer.evaluation(eval_data)
