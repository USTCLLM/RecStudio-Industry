import os
import time
import re
import importlib 
from collections import defaultdict
from collections.abc import Iterable
from abc import abstractmethod
import argparse
from copy import deepcopy 

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
import yaml
import json 
import redis 
import faiss

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class InferenceEngine(object):

    def __init__(self, config:dict) -> None:
        
        # load config 
        self.config = config
        with open(os.path.join(config['model_ckpt_path'], 'model_config.json'), 'r', encoding='utf-8') as f:
            self.model_ckpt_config = json.load(f)
        self.feature_config = deepcopy(self.model_ckpt_config['data_attr'])
        with open(config['feature_cache_config_path'], 'r') as f:
            self.feature_cache_config = yaml.safe_load(f)
        with open(config['retrieve_index_config_path'], 'r') as f:
            self.retrieve_index_config = yaml.safe_load(f)
        
        # load infer data 
        self.infer_df = pd.read_feather(config['inference_dataset_path'])

        # load candidates
        if config['stage'] == 'retrieve':
            if config['retrieval_mode'] == 'u2i':
                self.u2i_index = faiss.read_index(self.retrieve_index_config['u2i_index_path'])
                self.u2i_index.nprobe = self.retrieve_index_config['nprobe']
                self.item_ids_table = np.load(self.retrieve_index_config['item_ids_path'])
            elif config['retrieval_mode'] == 'i2i':
                self.i2i_redis_client = redis.Redis(host=self.config['i2i_redis_host'], 
                                    port=self.config['i2i_redis_port'], 
                                    db=self.config['i2i_redis_db'])

        else:
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

        # connect to redis for feature cache
        self.redis_client = redis.Redis(host=self.feature_cache_config['host'], 
                                        port=self.feature_cache_config['port'], 
                                        db=self.feature_cache_config['db'])
        
        # load model session
        if config['infer_mode'] == 'onnx':
            self.ort_session = self.get_ort_session()
            print(f'Session is using : {self.ort_session.get_providers()}')
        elif config['infer_mode'] == 'trt':
            self.engine = self.get_trt_session()

    @abstractmethod
    def get_ort_session(self) -> ort.InferenceSession:
        pass 
    
    def batch_inference_retrieval(self):
        # iterate infer data 
        infer_res = []
        num_batch = (len(self.infer_df) - 1) // self.config['infer_batch_size'] + 1 
        for batch_idx in tqdm(range(num_batch)):
            batch_st_time = time.time()
            batch_st = batch_idx * self.config['infer_batch_size']
            batch_ed = (batch_idx + 1) * self.config['infer_batch_size']
            batch_infer_df = self.infer_df.iloc[batch_st : batch_ed]

            # get user_context features 
            batch_user_context_dict = self.get_context_features(batch_infer_df)
            
            feed_dict = {}
            feed_dict.update(batch_user_context_dict)
            for key in feed_dict:
                feed_dict[key] = np.array(feed_dict[key])
            # print(feed_dict)

            if self.config['retrieval_mode'] == 'u2i':
                if self.config['infer_mode'] == 'onnx':
                    batch_user_embedding = self.ort_session.run(
                        output_names=["output"],
                        input_feed=feed_dict
                    )[0]
                elif self.config['infer_mode'] == 'trt':
                    batch_user_embedding = self.infer_with_trt(feed_dict)

                user_embedding_np = batch_user_embedding[:batch_infer_df.shape[0]]
                D, I = self.u2i_index.search(user_embedding_np, self.config['output_topk'])
                batch_outputs = I
            elif self.config['retrieval_mode'] == 'i2i':
                seq_video_ids = feed_dict['seq_video_id']
                batch_outputs = self.get_i2i_recommendations(seq_video_ids)
            
            print(batch_outputs.shape)
            batch_ed_time = time.time()
            print(f'batch time: {batch_ed_time - batch_st_time}s')
            infer_res.append(batch_outputs)
        all_res = np.concatenate(infer_res, axis=0)
        if self.config['retrieval_mode'] == 'u2i':
            return self.item_ids_table[all_res]
        else:
            return all_res
    
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
    
    def batch_inference(self, batch_candidates_df:pd.DataFrame=None):
        '''
        batch inference
        Args:
            batch_candidates_df: pd.DataFrame: candidates of the batch request. If None, get candidates from self.candidates_df.
        Returns:
            infer_res: np.ndarray
        '''
        infer_res = []
        num_batch = (len(self.infer_df) - 1) // self.config['infer_batch_size'] + 1 
        for batch_idx in tqdm(range(num_batch)):
            batch_st_time = time.time()
            batch_st = batch_idx * self.config['infer_batch_size']
            batch_ed = (batch_idx + 1) * self.config['infer_batch_size']
            batch_infer_df = self.infer_df.iloc[batch_st : batch_ed]
            if batch_candidates_df is None:
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
                output_names=["output"],
                input_feed=feed_dict
            )[0]
            batch_ed_time = time.time()
            print(f'batch time: {batch_ed_time - batch_st_time}s')
            infer_res.append(batch_outputs)
        return np.concatenate(infer_res, axis=0)

    def get_candidates(self, batch_infer_df:pd.DataFrame):

        '''
        Obtain a Batch of Candidate items

        Args:
        - `batch_infer_df` (pd.DataFrame): A DataFrame containing data to be inferred.

        Returns:
        - pd.DataFrame: A DataFrame containing candidate objects, with keys representing the features of the candidates.

        Notes:
        - Each row of `candidates_df` contains all features of candidates item of a key, formatted as `{feat1: [N], feat2: [N]}`.
        - Each row in `batch_infer_df` will generate a key based on the `request_features` field in its configuration.
        - This key is then used to find the corresponding candidate objects in `candidates_df`, and a DataFrame composed of these candidate objects is returned.
        '''
        # candidates_df : [keys: {feat1 : [N], feat2 : [N]}]
        batch_keys = batch_infer_df.apply(lambda row : '_'.join([str(row[feat]) for feat in self.config['request_features']]),
                                          axis='columns')
        batch_candidates = self.candidates_df.loc[batch_keys].reset_index(drop=True)
        return batch_candidates
    
    def get_user_context_features(self, batch_infer_df:pd.DataFrame): 
        '''
        get user and context features from redis
        Args:
            batch_infer_df: pd.DataFrame
        Returns:
            user_context_dict: dict
        '''
        # batch_infer_df : [B, M] 
        '''
        context_features:
        [
            "user_id",
            "device_id",
            "age",
            "gender",
            "province", 
            {"seq_effective_50" : ["video_id", "author_id", "category_level_two", "category_level_one", "upload_type"]}
        ]
        ''' 
        
        batch_infer_df = batch_infer_df.rename(mapper=(lambda col: col.strip(' ')), axis='columns')
        
        # user and context side features 
        context_features = [sub_feat for feat in self.feature_config['context_features'] 
                            for sub_feat in (list(feat.keys()) if isinstance(feat, dict) else [feat])]
        user_context_dict = self._row_get_features(
            batch_infer_df, 
            context_features, 
            [self.feature_cache_config['features'][feat] for feat in context_features])
        
        for feat in self.feature_config['context_features']:
            if isinstance(feat, dict):
                for feat_name, feat_fields in feat.items():
                    cur_dict = defaultdict(list) 
                    if isinstance(user_context_dict[feat_name][0], Iterable):
                        for seq in user_context_dict[feat_name]:
                            for field in feat_fields: 
                                cur_list = [getattr(proto, field) for proto in seq]
                                cur_dict[feat_name + '_' + field].append(cur_list)
                        user_context_dict.update(cur_dict)  
                        del user_context_dict[feat_name]
                    else:
                        for proto in user_context_dict[feat_name]:
                            for field in feat_fields:
                                cur_dict[feat_name + '_' + field].append(getattr(proto, field))
                        del user_context_dict[feat_name]
 
        return user_context_dict 

    def get_candidates_features(self, batch_candidates_df:pd.DataFrame):
        '''
        get candidates features from redis
        Args:
            batch_candidates_df: pd.DataFrame
        Returns:
            candidates_dict: dict
        '''
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

    def save_output_topk(self, output):
        output_df = {}
        output_df['_'.join(self.config['request_features'])] = \
            self.infer_df.apply(lambda row : '_'.join([str(row[feat]) for feat in self.config['request_features']]),
                                axis='columns')
        output_df[self.feature_config['fiid']] = output.tolist()
        output_df = pd.DataFrame(output_df)
        output_df.to_feather(self.config['output_save_path'])
    
    def build_engine(self, onnx_file_path, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # Parse ONNX model
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError('Failed to parse ONNX model')

            # Create builder config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

            # Create optimization profile
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_name = network.get_input(i).name
                input_shape = network.get_input(i).shape
                # Set the min, opt, and max shapes for the input
                min_shape = [1 if dim == -1 else dim for dim in input_shape]
                opt_shape = [self.config['infer_batch_size'] if dim == -1 else dim for dim in input_shape]
                max_shape = [self.config['infer_batch_size'] if dim == -1 else dim for dim in input_shape]
                profile.set_shape(input_name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
            config.add_optimization_profile(profile)

            # Build and serialize the engine
            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_file_path, 'wb') as f:
                f.write(serialized_engine)
            return serialized_engine

    def load_engine(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_trt_session(self):
        model_onnx_path = os.path.join(self.config['model_ckpt_path'], 'model_onnx.pb')
        trt_engine_path = os.path.join(self.config['model_ckpt_path'], 'model_trt.engine')

        # Set the GPU device
        cuda.Device(self.config['infer_device']).make_context()

        # Build or load the engine
        if not os.path.exists(trt_engine_path):
            serialized_engine = self.build_engine(model_onnx_path, trt_engine_path)
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:
            engine = self.load_engine(trt_engine_path)

        return engine
    
    def infer_with_trt(self, inputs):
        with self.engine.create_execution_context() as context:
            stream = cuda.Stream()
            bindings = [0] * self.engine.num_io_tensors

            input_memory = []
            output_buffers = {}
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    if -1 in tuple(self.engine.get_tensor_shape(tensor_name)):  # dynamic
                        context.set_input_shape(tensor_name, tuple(self.engine.get_tensor_profile_shape(tensor_name, 0)[2]))
                    input_mem = cuda.mem_alloc(inputs[tensor_name].nbytes)
                    bindings[i] = int(input_mem)
                    context.set_tensor_address(tensor_name, int(input_mem))
                    cuda.memcpy_htod_async(input_mem, inputs[tensor_name], stream)
                    input_memory.append(input_mem)
                else:  # output
                    shape = tuple(context.get_tensor_shape(tensor_name))
                    output_buffer = np.empty(shape, dtype=dtype)
                    output_buffer = np.ascontiguousarray(output_buffer)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings[i] = int(output_memory)
                    context.set_tensor_address(tensor_name, int(output_memory))
                    output_buffers[tensor_name] = (output_buffer, output_memory)

            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()

            for tensor_name, (output_buffer, output_memory) in output_buffers.items():
                cuda.memcpy_dtoh(output_buffer, output_memory)

        return output_buffers['output'][0]