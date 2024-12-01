import sys 
sys.path.append('.')

import gzip
import redis
from inference.feature_insert.protos import recflow_pb2
import numpy as np
import pandas as pd  
from tqdm import tqdm


# connect to Redis
r = redis.Redis(host='localhost', port=6379, db=1)

# Item
test_video_info = pd.read_feather('./inference/feature_data/recflow/realshow_test_video_info.feather')
for row in tqdm(test_video_info.itertuples(), total=len(test_video_info)):

    video_id = getattr(row, 'video_id')
    feat_list = []
    feat_list.append(getattr(row, 'video_id'))
    feat_list.append(getattr(row, 'author_id'))
    feat_list.append(getattr(row, '_3'))
    feat_list.append(getattr(row, 'upload_type'))
    feat_list.append(getattr(row, 'upload_timestamp'))
    feat_list.append(getattr(row, 'category_level_one'))
    feat_list = [str(feat) for feat in feat_list]
    
    r.set(f"recflow:item:{video_id}", ":".join(feat_list))
    

print("Item features are stored in Redis.")

# User
test_user_info = np.load('./inference/feature_data/recflow/test_user_info.npz')['arr_0']
for row in tqdm(test_user_info):

    feat_list = []
    feat_list.append(row[0])
    feat_list.append(row[1])
    feat_list.append(row[2])
    feat_list.append(row[3])
    feat_list.append(row[4])
    feat_list.append(row[5])
    feat_list.append(row[6])
    feat_list.append(list(map(int, row[7:])))
    feat_list = [str(feat) for feat in feat_list]

    # 3. 将压缩后的数据存储到 Redis 中
    r.set(f"recflow:user_timestamp:{row[1]}_{row[2]}", ":".join(feat_list))

print("UserTimestamp features are stored in Redis.")