import torch 
import numpy as np
import onnxruntime as ort
import argparse 
import yaml 
import pandas as pd 
import redis 
from collections import defaultdict
import re
from collections.abc import Iterable 
import numpy as np 
import importlib 
from tqdm import tqdm 
import time

class InferenceEngine(object):

    def __init__(self, config:dict) -> None:
        
        # load config 
        self.config = config
        with open(config['feature_config_path'], 'r') as f:
            self.feature_config = yaml.safe_load(f)
        with open(config['feature_cache_config_path'], 'r') as f:
            self.feature_cache_config = yaml.safe_load(f)
        
        # load infer data 
        self.infer_df = pd.read_feather(config['inference_dataset_path'])

        # load candidates
        self.candidates_df = pd.read_feather(config['candidates_path'])
        self.candidates_df = self.candidates_df.set_index('_'.join(config['request_features']))

        # load cache protos 
        self.key_temp2proto = {}
        for key_temp, proto_dict in self.feature_cache_config['key_temp2proto'].items():
            proto_spec = importlib.util.spec_from_file_location(
                f'{key_temp}_proto_module', proto_dict['module_path'])
            proto_module = importlib.util.module_from_spec(proto_spec)
            proto_spec.loader.exec_module(proto_module)
            self.key_temp2proto[key_temp] = getattr(proto_module, proto_dict['class_name'])

        # connect to redis 
        self.redis_client = redis.Redis(host=self.feature_cache_config['host'], 
                                        port=self.feature_cache_config['port'], 
                                        db=self.feature_cache_config['db'])
        
        # load model session
        self.ort_session = ort.InferenceSession(config['model_path'])
    
    def batch_inference(self):
        # iterate infer data 
        infer_res = []
        num_batch = (len(self.infer_df) - 1) // self.config['infer_batch_size'] + 1 
        for batch_idx in tqdm(range(num_batch)):
            batch_st_time = time.time()
            batch_st = batch_idx * self.config['infer_batch_size']
            batch_ed = (batch_idx + 1) * self.config['infer_batch_size']
            batch_infer_df = self.infer_df.iloc[batch_st : batch_ed]
            batch_candidates_df = self.get_candidates(batch_infer_df)

            # get user_context features 
            batch_user_context_dict = self.get_user_context_features(batch_infer_df)
            
            # get candidates features
            batch_candidates_dict = self.get_candidates_features(batch_candidates_df)
            # TODO: Cross Features

            feed_dict = {}
            feed_dict.update(batch_user_context_dict)
            feed_dict.update(batch_candidates_dict)
            feed_dict['output_topk'] = self.config['output_topk']
            for key in feed_dict:
                feed_dict[key] = np.array(feed_dict[key])
            batch_outputs = self.ort_session.run(
                output_names=["test_output"],
                input_feed=feed_dict
            )[0]
            batch_ed_time = time.time()
            print(f'batch time: {batch_ed_time - batch_st_time}s')
            infer_res.append(batch_outputs)
        return np.stack(infer_res, axis=0)

    def get_candidates(self, batch_infer_df:pd.DataFrame):
        # candidates_df : [keys: {feat1 : [B, N], feat2 : [B, N]}]
        batch_keys = batch_infer_df.apply(lambda row : '_'.join([str(row[feat]) for feat in self.config['request_features']]),
                                          axis='columns')
        batch_candidates = self.candidates_df.loc[batch_keys].reset_index(drop=True)
        return batch_candidates
    
    def get_user_context_features(self, batch_infer_df:pd.DataFrame): 
        # batch_infer_df : [B, M] 
        batch_infer_df = batch_infer_df.rename(mapper=(lambda col: col.strip(' ')), axis='columns')
        
        # user and context side features 
        user_context_dict = self._row_get_features(
            batch_infer_df, 
            self.feature_config['user_context_features'], 
            [self.feature_cache_config['features'][feat] for feat in self.feature_config['user_context_features']])
        return user_context_dict 

    def get_candidates_features(self, batch_candidates_df:pd.DataFrame):
        batch_candidates_df = batch_candidates_df.rename(mapper=(lambda col: col.strip(' ')), axis='columns')

        # candidates side features 
        # flatten candidates
        flattened_candidates_df = defaultdict(list) 
        num_candidates = len(batch_candidates_df.iloc[0, 0])
        for row in batch_candidates_df.itertuples():
            for col in batch_candidates_df.columns: 
                if not isinstance(getattr(row, col), np.ndarray):
                    raise ValueError('All elements of each columns of batch_candidates_df should be list')
                if num_candidates != len(getattr(row, col)):
                    raise ValueError('All features of one candidates should have equal length!')    
                flattened_candidates_df[col].extend(getattr(row, col).tolist()) 

        flattened_candidates_df = pd.DataFrame(flattened_candidates_df)
        # get flattened candidate features
        flattened_candidates_dict = self._row_get_features(
            flattened_candidates_df, 
            self.feature_config['item_features'], 
            [self.feature_cache_config['features'][feat] for feat in self.feature_config['item_features']])
        # fold candidate features
        candidates_dict = {}
        for key, value in flattened_candidates_dict.items():
            candidates_dict['candidates_' + key] = [value[i * num_candidates : (i + 1) * num_candidates] \
                                                    for i in range(len(batch_candidates_df))]

        return candidates_dict        

    def _row_get_features(self, row_df:pd.DataFrame, feats_list, feats_cache_list):
        # each row represents one entry 
        # row_df: [B, M]
        res_dict = defaultdict(list)
        # key_temp list 
        feats_key_temp_list = list(set([cache['key_temp'] for cache in feats_cache_list]))
        
        # get all keys and values related to these rows in one time 
        with self.redis_client.pipeline() as pipe:
            feats_all_key_and_temp = set()
            for row in row_df.itertuples():
                for key_temp in feats_key_temp_list:
                    key_feats = re.findall('{(.*?)}', key_temp) 
                    cur_key = key_temp
                    for key_feat in key_feats:
                        cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                    feats_all_key_and_temp.add((cur_key, key_temp))
            feats_all_key_and_temp = list(feats_all_key_and_temp)

            redis_st_time = time.time()
            for key, _ in feats_all_key_and_temp:
                pipe.get(key)
            feats_all_values = pipe.execute()
            redis_ed_time = time.time()
            print(f'redis time : {(redis_ed_time - redis_st_time)}s')
        
        parse_st_time = time.time()
        feats_k2p = {}
        for (key, key_temp), value in zip(feats_all_key_and_temp, feats_all_values):
            value_proto = self.key_temp2proto[key_temp]()
            value_proto.ParseFromString(value)
            feats_k2p[key] = value_proto
        parse_ed_time = time.time()
        print(f'parse time : {(parse_ed_time - parse_st_time)}s')

        # get feats from these values
        for row in row_df.itertuples():
            cur_all_values = dict()
            for key_temp in feats_key_temp_list:
                key_feats = re.findall('{(.*?)}', key_temp) 
                cur_key = key_temp
                for key_feat in key_feats:
                    cur_key = cur_key.replace(f'{{{key_feat}}}', str(getattr(row, key_feat)))
                cur_all_values[key_temp] = feats_k2p[cur_key]

            for feat, cache in zip(feats_list, feats_cache_list):
                res_dict[feat].append(getattr(cur_all_values[cache['key_temp']], cache['field']))
        
        return res_dict

