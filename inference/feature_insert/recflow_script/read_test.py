import sys 
sys.path.append('.')

import gzip
import redis
from inference.feature_insert.protos import recflow_pb2
import numpy as np
import pandas as pd  
from tqdm import tqdm
import time 


# connect to Redis
r_0 = redis.Redis(host='localhost', port=6379, db=0)
r_1 = redis.Redis(host='localhost', port=6379, db=1)


# 获取所有键
keys = r_0.keys()
# 累积计算每个键的内存占用
total_memory = 0
for key in tqdm(keys):
    total_memory += r_0.memory_usage(key)
# 输出总内存占用
print(f"Total memory used by db0: {total_memory} bytes")

# 获取所有键
keys = r_1.keys()
# 累积计算每个键的内存占用
total_memory = 0
for key in tqdm(keys):
    total_memory += r_1.memory_usage(key)
# 输出总内存占用
print(f"Total memory used by db1: {total_memory} bytes")

# Item
test_video_info = pd.read_feather('./inference/feature_data/recflow/realshow_test_video_info.feather')
random_video_id_list = list(test_video_info.iloc[np.random.randint(len(test_video_info), size=500000)]['video_id'])

# protobuffer 
read_time = 0 
parse_time = 0
for video_id in tqdm(random_video_id_list):
    read_st = time.time() 
    item_string = r_0.get(f'recflow:item:{video_id}')
    read_ed = time.time() 
    read_time += (read_ed - read_st)

    parse_st = time.time()
    item = recflow_pb2.Item()
    item.ParseFromString(item_string)

    # video_id = item.video_id
    # author_id = item.author_id
    # category_level_two = item.category_level_two
    # upload_type = item.upload_type
    # upload_timestamp = item.upload_timestamp
    # category_level_one = item.category_level_one

    parse_ed = time.time()
    parse_time += parse_ed - parse_st


print(f"item read time : {read_time}s, parse time : {parse_time}")



# baseline 
baseline_read_time = 0 
baseline_parse_time = 0
for video_id in tqdm(random_video_id_list):
    read_st = time.time() 
    item_string = r_1.get(f'recflow:item:{video_id}')
    read_ed = time.time() 
    baseline_read_time += (read_ed - read_st)


    parse_st = time.time()

    feat_list = item_string.decode('utf-8').split(':')

    # video_id = int(feat_list[0])
    # author_id = int(feat_list[1])
    # category_level_two = int(feat_list[2])
    # upload_type = int(feat_list[3])
    # upload_timestamp = int(feat_list[4])
    # category_level_one = int(feat_list[5])

    parse_ed = time.time()
    baseline_parse_time += parse_ed - parse_st

print(f"item baseline read time : {baseline_read_time}s, parse time : {baseline_parse_time}")



# user_timestamp
test_user_info = np.load('./inference/feature_data/recflow/test_user_info.npz')['arr_0']

random_idx = np.random.randint(len(test_user_info), size=500000)
random_user_timestamp_list = list(zip(test_user_info[random_idx][:, 1].tolist(), test_user_info[random_idx][:, 2].tolist()))

# protobuffer 
read_time = 0 
parse_time = 0
for user_id, timestamp in tqdm(random_user_timestamp_list):
    read_st = time.time() 
    user_string = r_0.get(f'recflow:user_timestamp:{user_id}_{timestamp}')
    read_ed = time.time() 
    read_time += (read_ed - read_st)

    parse_st = time.time()
    user_timestamp = recflow_pb2.UserTimestamp()
    user_timestamp.ParseFromString(user_string)
    # request_id = user_timestamp.request_id
    # user_id = user_timestamp.user_id
    # request_timestamp = user_timestamp.request_timestamp
    # device_id = user_timestamp.device_id
    # age = user_timestamp.age
    # gender = user_timestamp.gender
    # province = user_timestamp.province
    # seq_effective_50 = user_timestamp.seq_effective_50
    parse_ed = time.time()
    parse_time += parse_ed - parse_st

print(f"user_timestamp read time : {read_time}s, parse time : {parse_time}")


# baseline 
baseline_read_time = 0 
baseline_parse_time = 0
for user_id, timestamp in tqdm(random_user_timestamp_list):
    read_st = time.time() 
    user_string = r_1.get(f'recflow:user_timestamp:{user_id}_{timestamp}')
    read_ed = time.time() 
    baseline_read_time += (read_ed - read_st)

    parse_st = time.time()
    feat_list = user_string.decode('utf-8').split(':')
    # request_id = int(feat_list[0])
    # user_id = int(feat_list[1])
    # request_timestamp = int(feat_list[2])
    # device_id = int(feat_list[3])
    # age = int(feat_list[4])
    # gender = int(feat_list[5])
    # province = int(feat_list[6])
    # seq_effective_50 = eval(feat_list[7])
    parse_ed = time.time()
    baseline_parse_time += parse_ed - parse_st

print(f"user_timestamp baseline read time : {baseline_read_time}s, parse time : {baseline_parse_time}")


