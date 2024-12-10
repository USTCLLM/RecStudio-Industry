import sys 
sys.path.append('.')

import gzip
import redis
from inference.feature_insert.protos import recflow_pb2
import numpy as np
import pandas as pd  
from tqdm import tqdm
import yaml 

# connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Item
test_video_info = pd.read_feather('./inference/feature_data/recflow/realshow_test_video_info.feather')
for row in tqdm(test_video_info.itertuples(), total=len(test_video_info)):

    # 0. Create a message object
    item = recflow_pb2.Item()
    item.video_id = getattr(row, 'video_id')
    item.author_id = getattr(row, 'author_id')
    item.category_level_two = getattr(row, '_3')
    item.upload_type = getattr(row, 'upload_type')
    item.upload_timestamp = getattr(row, 'upload_timestamp')
    item.category_level_one = getattr(row, 'category_level_one')
    
    # 1. Serialize the Protobuf object into binary data
    serialized_data = item.SerializeToString()

    # 2. Store the compressed data in Redis
    r.set(f"recflow:item:{item.video_id}", serialized_data)
    

print("Item features are stored in Redis.")

# User
test_user_info = np.load('./inference/feature_data/recflow/test_user_info.npz')['arr_0']
for row in tqdm(test_user_info):

    # 0. Create a message object 
    user_timestamp = recflow_pb2.UserTimestamp()
    user_timestamp.request_id = row[0]
    user_timestamp.user_id = row[1]
    user_timestamp.request_timestamp = row[2]
    user_timestamp.device_id = row[3]
    user_timestamp.age = row[4]
    user_timestamp.gender = row[5]
    user_timestamp.province = row[6]
    
    for behavior in np.split(row[7:], len(row[7:]) // 6):
        item = user_timestamp.seq_effective_50.add()
        item.video_id = behavior[0]
        item.author_id = behavior[1]
        item.category_level_two = behavior[2]
        item.category_level_one = behavior[3]
        item.upload_type = behavior[4]
        item.request_timestamp = behavior[5]

    # 1. Serialize the Protobuf object into binary data
    serialized_data = user_timestamp.SerializeToString()

    # 2. Store the compressed data in Redis
    r.set(f"recflow:user_timestamp:{row[1]}_{row[2]}", serialized_data)

print("UserTimestamp features are stored in Redis.")

# create feature cache config 
final_dict = {}
final_dict['host'] = r.get_connection_kwargs()['host']
final_dict['port'] = r.get_connection_kwargs()['port']
final_dict['db'] = r.get_connection_kwargs()['db']
feat_dict = {}
for col in test_video_info.columns:
    feat_dict[col.strip(' ')] = {}
    feat_dict[col.strip(' ')]['key_temp'] = 'recflow:item:{video_id}'
    feat_dict[col.strip(' ')]['field'] = col.strip(' ')
for col in ['request_id', 'user_id', 'request_timestamp', 'device_id', 'age', 'gender', 'province', 'seq_effective_50']:
    feat_dict[col.strip(' ')] = {}
    feat_dict[col.strip(' ')]['key_temp'] = 'recflow:user_timestamp:{user_id}_{request_timestamp}'
    feat_dict[col.strip(' ')]['field'] = col.strip(' ')
final_dict['features'] = feat_dict 
final_dict['key_temp2proto'] = {
    'recflow:item:{video_id}' : {'class_name': 'Item', 'module_path': recflow_pb2.__file__},
    'recflow:user_timestamp:{user_id}_{request_timestamp}' : {'class_name': 'UserTimestamp', 'module_path': recflow_pb2.__file__}
}

# save dict as yaml 
with open('./inference/feature_insert/feature_cache_configs/recflow_feature_cache_config.yaml', 'w') as file:
    yaml.dump(final_dict, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
