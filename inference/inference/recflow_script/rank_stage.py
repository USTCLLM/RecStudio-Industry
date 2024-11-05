import os 
import sys
sys.path.append('.')
import argparse 
from collections import defaultdict
import time 

from pandas import DataFrame
import yaml
import json 
import torch 
import onnx 
import onnxruntime as ort
from tqdm import tqdm 
import numpy as np 

from inference.inference.inference_engine import InferenceEngine
from rs4industry.model.rankers import MLPRanker
from rs4industry.data.dataset import get_datasets

class RankerInferenceEngine(InferenceEngine):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # put seq into context_features
        # self.feature_config is deepcopy of self.model_ckpt_config['data_attr']
        if 'seq_features' in self.model_ckpt_config['data_attr']:
            self.feature_config['context_features'].append({'seq_effective_50' : self.model_ckpt_config['data_attr']['seq_features']})
    
    # TODO: move this function to train
    def get_ort_session(self):
        
        # TODO: how to get data_config from self.model_ckpt_config['data_attr']
        data_config_path = "/data1/home/recstudio/huangxu/rec-studio-industry/examples/config/data/recflow_ranker.json"
        (train_data, eval_data), data_config = get_datasets(data_config_path)

        model = MLPRanker(data_config, os.path.join(self.config['model_ckpt_path'], 'model_config.json'))
        checkpoint = torch.load(os.path.join(self.config['model_ckpt_path'], 'model.pt'),
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.eval()
        model.forward = model.predict 

        input_names = []
        dynamic_axes = {} 

        # user/context input
        context_input = {}
        for feat in self.feature_config['context_features']:
            if isinstance(feat, str):
                context_input[feat] = torch.randint(self.feature_config['stats'][feat], (5,))
                input_names.append(feat)
                dynamic_axes[feat] = {0: "batch_size"}
        # TODO: need to be modified for better universality 
        # special process for seq feature 
        # TODO: how to get the length of the seq 
        seq_input = {}
        for field in self.feature_config['seq_features']:
            seq_input[field] = torch.randint(self.feature_config['stats'][field], (5, 50))
            input_names.append('seq_' + field)
            dynamic_axes['seq_' + field] = {0: "batch_size"}
        context_input['seq'] = seq_input
                        
        # candidates input 
        candidates_input = {} 
        for feat in self.feature_config['item_features']:
            if isinstance(feat, str):
                candidates_input[feat] = torch.randint(self.feature_config['stats'][feat], (5, 16))
                input_names.append('candidates_' + feat)
                dynamic_axes['candidates_' + feat] = {0: "batch_size", 1: "num_candidates"}

        output_topk = self.config['output_topk']
        input_names.append('output_topk')

        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        torch.onnx.export(
            model,
            (context_input, candidates_input, output_topk), 
            model_onnx_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=15,
            verbose=True
        )
        
        # print graph
        onnx_model = onnx.load(model_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("=" * 25 + 'comp graph : ' + "=" * 25)
        print(onnx.helper.printable_graph(onnx_model.graph))

        if self.config['infer_device'] == 'cpu':
            providers = ["CPUExecutionProvider"]
        elif isinstance(self.config['infer_device'], int):
            providers = [("CUDAExecutionProvider", {"device_id": self.config['infer_device']})]
        return ort.InferenceSession(model_onnx_path, providers=providers)
    
    # TODO: model should use seq_effective_50 as name of seq feature
    def get_context_features(self, batch_infer_df: DataFrame):
        batch_user_context_dict = super().get_context_features(batch_infer_df)
        
        for k, v in list(batch_user_context_dict.items()):
            if k.startswith('seq_effective_50_'):
                batch_user_context_dict['seq_' + k[len('seq_effective_50_'):]] = v
                del batch_user_context_dict[k]
        
        return batch_user_context_dict 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_config_path", type=str, required=True, help="Inference config file")  
    args = parser.parse_args()

    with open(args.infer_config_path, 'r') as f:
        config = yaml.safe_load(f)

    rank_inference_engine = RankerInferenceEngine(config)
    ranker_outputs = rank_inference_engine.batch_inference()
    rank_inference_engine.save_output_topk(ranker_outputs)