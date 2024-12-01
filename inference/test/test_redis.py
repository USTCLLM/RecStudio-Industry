import gzip
import redis
from protos import recflow_pb2
import numpy as np
import pandas as pd  
from tqdm import tqdm


# connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# get value from redis 
item_string = r.get('item:1155335')

item = recflow_pb2.Item()
item.ParseFromString(item_string)
print(item)


# get value from redis 
user_timestamp_string = r.get("user_timestamp:1635_1708185854977")

user_timestamp = recflow_pb2.UserTimestamp()
user_timestamp.ParseFromString(user_timestamp_string)
print(user_timestamp)

print('cxl')