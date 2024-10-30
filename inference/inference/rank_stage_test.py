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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict(context_input:dict, candidates_input:dict, output_topk:int):
    # context_input: dict 
    # candidates_input: dict
    # output_topk: int
    user_id = context_input['user_id'] % 10000 # [B]
    request_timestamp = context_input['request_timestamp'] * 2 % 10000 # [B]
    device_id = context_input['device_id'] * 3 % 10000 # [B]
    age = context_input['age'] * 4 % 10000 # [B]
    gender = context_input['gender'] * 5 % 10000 # [B]
    seq_effective_50 = context_input['seq_effective_50'] * 6 % 10000 # [B, 6 * L]
    
    video_id = candidates_input['candidates_video_id'] * 7 % 10000 # [B, N]
    author_id = candidates_input['candidates_author_id'] * 8 % 10000 # [B, N]
    category_level_two = candidates_input['candidates_category_level_two'] * 9 % 10000 # [B, N]
    upload_type = candidates_input['candidates_upload_type'] * 10 % 10000 # [B, N]
    upload_timestamp = candidates_input['candidates_upload_timestamp'] * 11 % 10000 # [B, N]
    category_level_one = candidates_input['candidates_category_level_one'] * 12 % 10000 # [B, N]

    range_t = torch.arange(100)
    
    return range_t.topk(output_topk).values.sum() + torch.sum(user_id) \
        + torch.sum(request_timestamp) \
        + torch.sum(device_id) \
        + torch.sum(age) \
        + torch.sum(gender) \
        + torch.sum(seq_effective_50) \
        + torch.sum(video_id) \
        + torch.sum(author_id) \
        + torch.sum(category_level_two) \
        + torch.sum(upload_type) \
        + torch.sum(upload_timestamp) \
        + torch.sum(category_level_one) 

def _row_get_features(redis_client:redis.Redis, row_df:pd.DataFrame, feats_list, feats_cache_list, 
                      key_temp2proto:dict):
    # each row represents one entry 
    # row_df: [B, M]
    res_dict = defaultdict(list)
    # key_temp list 
    feats_key_temp_list = list(set([cache['key_temp'] for cache in feats_cache_list]))
    
    # get all keys and values related to these rows in one time 
    with redis_client.pipeline() as pipe:
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
        value_proto = key_temp2proto[key_temp]()
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

def get_features(redis_client:redis.Redis, batch_infer_df:pd.DataFrame, batch_candidates_df:pd.DataFrame, 
                 feature_config, feature_cache_config, key_temp2proto:dict):
    # batch_infer_df : [B, M] 
    # batch_candidates_df : {feat1 : [B, N], feat2 : [B, N]}

    batch_infer_df = batch_infer_df.rename(mapper=(lambda col: col.strip(' ')), axis='columns')
    batch_candidates_df = batch_candidates_df.rename(mapper=(lambda col: col.strip(' ')), axis='columns')

    # user and context side features 
    user_context_dict = _row_get_features(redis_client, batch_infer_df, 
                                      feature_config['user_context_features'], 
                                      [feature_cache_config[feat] for feat in feature_config['user_context_features']],
                                      key_temp2proto)
    
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
    flattened_candidates_dict = _row_get_features(redis_client, flattened_candidates_df, 
                                        feature_config['item_features'], 
                                        [feature_cache_config[feat] for feat in feature_config['item_features']],
                                        key_temp2proto)
    # fold candidate features
    candidates_dict = {}
    for key, value in flattened_candidates_dict.items():
        candidates_dict['candidates_' + key] = [value[i * num_candidates : (i + 1) * num_candidates] \
                                                for i in range(len(batch_candidates_df))]
        
    return user_context_dict, candidates_dict

def get_candidates(request_features:str, batch_infer_df:pd.DataFrame, candidates_df:pd.DataFrame):
    # candidates_df : [keys: {feat1 : [B, N], feat2 : [B, N]}]
    batch_keys = batch_infer_df.apply(lambda row : '_'.join([str(row[feat]) for feat in request_features]), axis='columns')
    batch_candidates = candidates_df.loc[batch_keys].reset_index(drop=True)
    return batch_candidates
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_config_path", type=str, required=True, help="Inference config file")  
    args = parser.parse_args()

    # load config 
    with open(args.infer_config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(config['feature_config_path'], 'r') as f:
        feature_config = yaml.safe_load(f)
    with open(config['feature_cache_config_path'], 'r') as f:
        feature_cache_config = yaml.safe_load(f)

    # load infer data 
    infer_df = pd.read_feather(config['inference_dataset_path'])

    # load candidates
    candidates_df = pd.read_feather(config['candidates_path'])
    candidates_df = candidates_df.set_index('_'.join(config['request_features']))

    # load cache protos 
    key_temp2proto = {}
    for key_temp, proto_dict in feature_cache_config['key_temp2proto'].items():
        proto_spec = importlib.util.spec_from_file_location(
            f'{key_temp}_proto_module', proto_dict['module_path'])
        proto_module = importlib.util.module_from_spec(proto_spec)
        proto_spec.loader.exec_module(proto_module)
        key_temp2proto[key_temp] = getattr(proto_module, proto_dict['class_name'])

    # connect to redis 
    redis_client = redis.Redis(host=feature_cache_config['host'], 
                               port=feature_cache_config['port'], 
                               db=feature_cache_config['db'])
    
    ort_session = ort.InferenceSession(config['model_path'])

    # iterate infer data 
    num_batch = (len(infer_df) - 1) // config['infer_batch_size'] + 1 
    for batch_idx in tqdm(range(num_batch)):
        batch_st_time = time.time()
        batch_st = batch_idx * config['infer_batch_size']
        batch_ed = (batch_idx + 1) * config['infer_batch_size']
        batch_infer_df = infer_df.iloc[batch_st : batch_ed]
        batch_candidates_df = get_candidates(config['request_features'], 
                                             batch_infer_df, candidates_df)
        batch_user_context_dict, batch_candidates_dict = get_features(
            redis_client, batch_infer_df, batch_candidates_df,
            feature_config, feature_cache_config['features'], 
            key_temp2proto)
        
        feed_dict = {}
        feed_dict.update(batch_user_context_dict)
        feed_dict.update(batch_candidates_dict)
        feed_dict['output_topk'] = config['output_topk']
        for key in feed_dict:
            feed_dict[key] = np.array(feed_dict[key])
        batch_outputs = ort_session.run(
            output_names=None,
            input_feed=feed_dict
        )
        print(f"ONNX model's answer: {batch_outputs[0]}")
        batch_ed_time = time.time()
        print(f'batch time: {batch_ed_time - batch_st_time}s')



    

