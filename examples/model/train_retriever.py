from rs4industry.trainer import Trainer
from rs4industry.data.dataset import get_datasets
from rs4industry.model.retrievers import MLPRetriever

data_config_path = "../config/data/recflow_retriever.json"
train_config_path = "../config/mlp_retriever/train.json"
model_config_path = "../config/mlp_retriever/model.json"

(train_data, eval_data), data_config = get_datasets(data_config_path)

model = MLPRetriever(data_config, model_config_path)

trainer = Trainer(model, train_config_path)

trainer.fit(train_data, eval_data)

