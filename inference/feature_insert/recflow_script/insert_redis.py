import sys 
sys.path.append('.')

import gzip
import redis
from inference.feature_insert.protos import recflow_pb2
import numpy as np
import pandas as pd  
from tqdm import tqdm


# connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Item
test_video_info = pd.read_feather('./inference/feature_data/recflow/realshow_test_video_info.feather')
for row in tqdm(test_video_info.itertuples(), total=len(test_video_info)):

    # 0. 创建 message 对象
    item = recflow_pb2.Item()
    item.video_id = getattr(row, 'video_id')
    item.author_id = getattr(row, 'author_id')
    item.category_level_two = getattr(row, '_3')
    item.upload_type = getattr(row, 'upload_type')
    item.upload_timestamp = getattr(row, 'upload_timestamp')
    item.category_level_one = getattr(row, 'category_level_one')
    
    # 1. 序列化 Protobuf 对象为二进制数据
    serialized_data = item.SerializeToString()

    # 2. 使用 gzip 压缩序列化后的数据
    # compressed_data = gzip.compress(serialized_data)

    # 3. 将压缩后的数据存储到 Redis 中
    r.set(f"recflow:item:{item.video_id}", serialized_data)
    

print("Item features are stored in Redis.")

# User
test_user_info = np.load('./inference/feature_data/recflow/test_user_info.npz')['arr_0']
for row in tqdm(test_user_info):

    # 0. 创建 message 对象
    user_timestamp = recflow_pb2.UserTimestamp()
    user_timestamp.request_id = row[0]
    user_timestamp.user_id = row[1]
    user_timestamp.request_timestamp = row[2]
    user_timestamp.device_id = row[3]
    user_timestamp.age = row[4]
    user_timestamp.gender = row[5]
    user_timestamp.province = row[6]
    user_timestamp.seq_effective_50.extend(list(row[7:]))
    
    # 1. 序列化 Protobuf 对象为二进制数据
    serialized_data = user_timestamp.SerializeToString()

    # 2. 使用 gzip 压缩序列化后的数据
    # compressed_data = gzip.compress(serialized_data)

    # 3. 将压缩后的数据存储到 Redis 中
    r.set(f"recflow:user_timestamp:{row[1]}_{row[2]}", serialized_data)

print("UserTimestamp features are stored in Redis.")