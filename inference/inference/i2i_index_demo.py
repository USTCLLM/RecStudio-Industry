import numpy as np
import faiss
import redis
import torch
from tqdm import tqdm

file_path = '/data1/home/recstudio/wuchao/saved_model_demo/retrieval_best_ckpt/item_vectors.pt'
data = torch.load(file_path,map_location=torch.device('cpu'))

item_vectors = data['item_vectors']  # NxD
item_ids = data['item_ids']  # N
item_embeddings = item_vectors.numpy()
item_ids = item_ids.numpy()
print(item_ids.max())


print(item_embeddings.shape)
index_path = '/data1/home/recstudio/wuchao/saved_model_demo/faiss_item_ivf2.index'
index = faiss.read_index(index_path)
print('Read index Done')
k = 10
D, I = index.search(item_embeddings, k)
print('Search Done.')

redis_host = 'localhost'
redis_port = 6379
redis_db = 4
r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

for item_idx in tqdm(range(len(item_ids))):
    top10_indices = I[item_idx]
    top10_item_ids = item_ids[top10_indices].tolist()
    r.set(f'item:{item_ids[item_idx]}', ','.join(map(str, top10_item_ids)))

print("Top 10 item IDs for each item have been stored in Redis.")