import numpy as np
import pandas as pd  
import gzip 
import redis

test_video_info = pd.read_feather('./inference/feature_data/recflow/test_video_info.feather')

test_user_info = np.load('./inference/feature_data/recflow/test_user_info.npz')['arr_0']


print('cxl')