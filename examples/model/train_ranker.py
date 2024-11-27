from rs4industry.data.dataset import get_datasets
# from rs4industry.model.retrievers import MLPRetriever
from rs4industry.model.rankers import MLPRanker
from rs4industry.trainer import Trainer

data_config_path = "/data1/home/recstudio/huangxu/rec-studio-industry/examples/config/data/recflow_ranker_mini.json"
train_config_path = "/data1/home/recstudio/huangxu/rec-studio-industry/examples/config/mlp_ranker/train.json"
model_config_path = "/data1/home/recstudio/huangxu/rec-studio-industry/examples/config/mlp_ranker/model.json"

(train_data, eval_data), data_config = get_datasets(data_config_path)

model = MLPRanker(data_config, model_config_path)

trainer = Trainer(model, train_config_path)

trainer.fit(train_data, eval_data)

