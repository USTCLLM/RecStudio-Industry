import os 
import pandas as pd

hdfs_url = 'hdfs://node1:8020/'
fpath_to_parquet = '/recstudio/recflow/realshow/2024-02-18.feather'
df = pd.read_feather(f"{hdfs_url}{fpath_to_parquet}")
df = df.drop_duplicates(subset=['user_id', 'request_timestamp'], ignore_index=True)
df = df[['request_id', 'user_id', 'request_timestamp']]

infer_data_dir = 'inference/inference_data/recflow'
df.to_feather(os.path.join(infer_data_dir, 'recflow_infer_data.feather'))

print(df)