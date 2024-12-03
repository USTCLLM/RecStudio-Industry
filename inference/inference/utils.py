import numpy as np
import faiss
import redis
import torch
from tqdm import tqdm

def gen_item_index(file_path, item_index_path, item_ids_path):
    # file_path = '/data1/home/recstudio/huangxu/saves/mlp_retriever_test/best_ckpt/item_vectors.pt'
    data = torch.load(file_path, map_location=torch.device('cpu'))

    item_vectors = data['item_vectors']  # NxD
    item_ids = data['item_ids']  # N
    item_embeddings = item_vectors.numpy()
    embedding_dim = item_embeddings.shape[1]
    item_num = item_embeddings.shape[0]
    print(item_num, embedding_dim)

    print('Constructing Index')
    quantizer = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100, faiss.METRIC_INNER_PRODUCT)
    index.train(item_embeddings)
    index.add(item_embeddings)

    # save index and item_ids 
    faiss.write_index(index, item_index_path)
    np.save(item_ids_path, item_ids.numpy())


def gen_i2i_index(topk, file_path, redis_host, redis_port, redis_db, item_index_path):
    data = torch.load(file_path, map_location=torch.device('cpu'))

    item_vectors = data['item_vectors']  # NxD
    item_ids = data['item_ids']  # N
    item_embeddings = item_vectors.numpy()
    item_ids = item_ids.numpy()
    print(item_ids.max())
    print(item_embeddings.shape)

    item_index = faiss.read_index(item_index_path)
    print('Read index Done')
    D, I = item_index.search(item_embeddings, topk)
    print('Search Done.')

    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

    for item_idx in tqdm(range(len(item_ids))):
        top10_indices = I[item_idx]
        top10_item_ids = item_ids[top10_indices].tolist()
        r.set(f'item:{item_ids[item_idx]}', ','.join(map(str, top10_item_ids)))

    print(f"Top {topk} item IDs for each item have been stored in Redis.")