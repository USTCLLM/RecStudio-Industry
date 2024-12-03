import os 
import pandas as pd
import numpy as np 
from tqdm import tqdm
np.random.seed(2024)

request_features = ['user_id', 'request_timestamp']
fpath_to_infer = 'inference/inference_data/recflow/recflow_infer_data.feather'
fpath_to_item = 'inference/feature_data/recflow/realshow_test_video_info.feather'
infer_df = pd.read_feather(fpath_to_infer)
item_df = pd.read_feather(fpath_to_item)


all_item_ids = np.array(item_df['video_id'])


request_key_list = []
candidates_list = []
for row in tqdm(infer_df.iloc.itertuples(), total=len(infer_df)):
    request_key_list.append('_'.join([str(getattr(row, feat)) for feat in request_features]))
    candidates_list.append(all_item_ids[np.random.choice(len(all_item_ids), 100)].tolist())


candidates_df = pd.DataFrame({'_'.join(request_features) : request_key_list, 'video_id': candidates_list})
candidates_demo_dir = 'inference/inference_data/recflow/'
candidates_df.to_feather(os.path.join(candidates_demo_dir, 'candidates_demo.feather'))
candidates_df = pd.read_feather(os.path.join(candidates_demo_dir, 'candidates_demo.feather'))
print('cxl')