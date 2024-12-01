import yaml

from rs4industry.data.dataset import get_datasets
from rs4industry.model.retrievers import MLPRetriever
from rs4industry.model.rankers import MLPRanker
from rs4industry.trainer import Trainer

from inference.inference.inference_engine import InferenceEngine

###train retriever###
retriever_data_config_path = "./examples/config/data/recflow_retriever.json"
retriever_train_config_path = "./examples/config/mlp_retriver/train.json"
retriever_model_config_path = "./examples/config/mlp_retriver/model.json"

(retriever_train_data, retriever_eval_data), retriever_data_config = get_datasets(retriever_data_config_path)

retriever_model = MLPRetriever(retriever_data_config, retriever_model_config_path)

retriever_trainer = Trainer(retriever_model, retriever_train_config_path)

retriever_trainer.fit(retriever_train_data, retriever_eval_data)

###train ranker###
ranker_data_config_path = "./examples/config/data/recflow_ranker.json"
ranker_train_config_path = "./examples/config/mlp_ranker/train.json"
ranker_model_config_path = "./examples/config/mlp_ranker/model.json"

(ranker_train_data, ranker_eval_data), ranker_data_config = get_datasets(ranker_data_config_path)

ranker_model = MLPRanker(ranker_data_config, ranker_model_config_path)

ranker_trainer = Trainer(ranker_model, ranker_train_config_path)

ranker_trainer.fit(ranker_train_data, ranker_eval_data)

###inference retriever###
infer_retrieval_path="./inference/inference/recflow_script/recflow_infer_retrieval.yaml"
with open(infer_retrieval_path, 'r') as f:
    retrieval_config = yaml.safe_load(f)

retrieval_inference_engine = InferenceEngine(retrieval_config)

retrieval_outputs = retrieval_inference_engine.batch_inference()

retrieval_inference_engine.save_output_topk(retrieval_outputs)


###inference retriever###
infer_ranker_path="./inference/inference/recflow_script/recflow_infer_config.yaml"
with open(infer_ranker_path, 'r') as f:
    ranker_config = yaml.safe_load(f)

rank_inference_engine = InferenceEngine(ranker_config)

ranker_outputs = rank_inference_engine.batch_inference()

rank_inference_engine.save_output_topk(ranker_outputs)