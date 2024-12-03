import numpy as np
import faiss
import torch
file_path = '/data1/home/recstudio/huangxu/saves/mlp_retriever_test/best_ckpt/item_vectors.pt'
data = torch.load(file_path,map_location=torch.device('cpu'))

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
index_path = '/data1/home/recstudio/wuchao/saved_model_demo/faiss_item_ivf2.index'
faiss.write_index(index, index_path)
item_ids_path = '/data1/home/recstudio/wuchao/saved_model_demo/item_ids.npy'
np.save(item_ids_path,item_ids.numpy())