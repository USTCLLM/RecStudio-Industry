import os
import sys
sys.path.append('.')
import argparse
import time

from pandas import DataFrame
import yaml
import torch
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from inference.inference.inference_engine import InferenceEngine
from inference.inference.utils import gen_item_index, gen_i2i_index
from rs4industry.model.base import BaseModel

import faiss
import redis

class RetrieverInferenceEngine(InferenceEngine):

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # put seq into context_features
        # self.feature_config is deepcopy of self.model_ckpt_config['data_attr']
        if 'seq_features' in self.model_ckpt_config['data_attr']:
            self.feature_config['context_features'].append({'seq_effective_50': self.model_ckpt_config['data_attr']['seq_features']})

        self.retrieve_index_config = config['retrieve_index_config']

        if config['stage'] == 'retrieve':
            if self.retrieve_index_config.get('gen_item_index', True):
                gen_item_index(os.path.join(config['model_ckpt_path'], 'item_vectors.pt'), 
                               self.retrieve_index_config['item_index_path'],
                               self.retrieve_index_config['item_ids_path'])
            if config['retrieval_mode'] == 'u2i':
                self.item_index = faiss.read_index(self.retrieve_index_config['item_index_path'])
                self.item_index.nprobe = self.retrieve_index_config['nprobe']
                self.item_ids_table = np.load(self.retrieve_index_config['item_ids_path'])
            elif config['retrieval_mode'] == 'i2i':
                if self.retrieve_index_config.get('gen_i2i_index', True):
                    gen_i2i_index(config['output_topk'],
                                  config['model_ckpt_path'], 
                                  self.retrieve_index_config['i2i_redis_host'],
                                  self.retrieve_index_config['i2i_redis_port'],
                                  self.retrieve_index_config['i2i_redis_db'],
                                  item_index_path=self.retrieve_index_config['item_index_path'])
                self.i2i_redis_client = redis.Redis(host=self.retrieve_index_config['i2i_redis_host'], 
                                    port=self.retrieve_index_config['i2i_redis_port'], 
                                    db=self.retrieve_index_config['i2i_redis_db'])
                
    def batch_inference(self, batch_infer_df:DataFrame):
        """
        Perform batch inference for a given batch of data.
        Args:
            batch_infer_df (DataFrame): A pandas DataFrame containing the batch of data to perform inference on.

        Returns:
            (np.ndarray):
                shape [batch_size, output_topk], the recommended items based on the provided inputs.
                - If retrieval_mode is 'u2i', returns a list of item IDs corresponding to the top-k recommendations for each user.
                - If retrieval_mode is 'i2i', returns a list of lists of item IDs representing the recommendations for each sequence of video IDs.
        """
        # iterate infer data 
        batch_st_time = time.time()

        # get user_context features 
        batch_user_context_dict = self.get_user_context_features(batch_infer_df)
        
        feed_dict = {}
        feed_dict.update(batch_user_context_dict)
        for key in feed_dict:
            feed_dict[key] = np.array(feed_dict[key])

        if self.config['retrieval_mode'] == 'u2i':
            if self.config['infer_mode'] == 'ort':
                batch_user_embedding = self.ort_session.run(
                    output_names=["output"],
                    input_feed=feed_dict
                )[0]
            elif self.config['infer_mode'] == 'trt':
                batch_user_embedding = self.infer_with_trt(feed_dict)

            user_embedding_np = batch_user_embedding[:batch_infer_df.shape[0]]
            D, I = self.item_index.search(user_embedding_np, self.config['output_topk'])
            batch_outputs = I
        elif self.config['retrieval_mode'] == 'i2i':
            seq_video_ids = feed_dict['seq_video_id']
            batch_outputs = self.get_i2i_recommendations(seq_video_ids)
        
        # print(batch_outputs.shape)
        batch_ed_time = time.time()
        # print(f'batch time: {batch_ed_time - batch_st_time}s')
      
        if self.config['retrieval_mode'] == 'u2i':
            return self.item_ids_table[batch_outputs]
        else:
            return batch_outputs
    
    def get_i2i_recommendations(self, seq_video_ids_batch):
        pipeline = self.i2i_redis_client.pipeline()
        for seq_video_ids in seq_video_ids_batch:
            for video_id in seq_video_ids:
                redis_key = f'item:{video_id}'
                pipeline.get(redis_key)
        results = pipeline.execute()

        all_top10_items = []

        for result in results:
            if result:
                top10_items = result.decode('utf-8').split(',')
                all_top10_items.extend(top10_items)
            else:
                print('Redis returned None for a key')

        all_top10_items = list(map(int, all_top10_items))
        all_top10_items = np.array(all_top10_items).reshape(seq_video_ids_batch.shape[0], -1)

        return all_top10_items

    def convert_to_onnx(self):
        model = BaseModel.from_pretrained(self.config['model_ckpt_path'])
        checkpoint = torch.load(os.path.join(self.config['model_ckpt_path'], 'model.pt'),
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.eval()
        model.forward = model.encode_query

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

        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        torch.onnx.export(
            model,
            (context_input, {}), 
            model_onnx_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=15,
            verbose=True
        )
        
    def get_user_context_features(self, batch_infer_df: DataFrame):
        batch_user_context_dict = super().get_user_context_features(batch_infer_df)

        for k, v in list(batch_user_context_dict.items()):
            if k.startswith('seq_effective_50_'):
                batch_user_context_dict['seq_' + k[len('seq_effective_50_'):]] = v
                del batch_user_context_dict[k]

        return batch_user_context_dict


if __name__ == '__main__':
    import pandas as pd
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_config_path", type=str, required=True, help="Inference config file")  
    args = parser.parse_args()

    with open(args.infer_config_path, 'r') as f:
        config = yaml.safe_load(f)

    retriever_inference_engine = RetrieverInferenceEngine(config)
    infer_df = pd.read_feather('inference/inference_data/recflow/recflow_infer_data.feather')
    for batch_idx in range(10):
        print(f"This is batch {batch_idx}")
        batch_st = batch_idx * 128 
        batch_ed = (batch_idx + 1) * 128 
        batch_infer_df = infer_df.iloc[batch_st:batch_ed]
        retriever_outputs = retriever_inference_engine.batch_inference(batch_infer_df)
        print(type(retriever_outputs), retriever_outputs.shape)
    if retriever_inference_engine.config['infer_mode'] == 'trt':
        cuda.Context.pop()
