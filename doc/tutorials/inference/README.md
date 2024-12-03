# Inference Tutorials
In an online recommendation system, handling a single request typically involves the following steps:
- **Receiving the request header**: The request header includes the user ID and context-specific features (e.g., location and timestamp of the request).
- **Obtaining the Candidate Item Set**: At each stage, the recommendation model receives the candidate item set from the previous stage (for the retrieval model, it is the entire item pool).
- **Retrieving Features**: At each stage, the system retrieves user- and item-related features required by the recommendation model based on the user ID and candidate item IDs. To enable fast access, user and item features are stored in a cache database (e.g., Redis) in a key-value format.
- **Sorting the Candidate Item Set**: At each stage, the recommendation model ranks the candidate items using the retrieved features and selects the top-k items to pass to the next stage (for the final stage, the top-k items are directly presented to the user).

## Storing Features in Cache Database
### Defining message in protobuf
To reduce the cache size occupied by features, Protobuf is used to serialize the features before storing them in the cache database. To use Protobuf,  message data structures must first be defined.

In the .proto file, the user and item message data structures are defined. For example, in recflow.proto:

Each feature of user and item is treated as a field of the message structure.

```protobuf
syntax = "proto3"; // the version of protobuf

package example;

message Item {
  int64 video_id = 1;
  int64 author_id = 2;
  int64 category_level_two = 3;
  int64 upload_type = 4;
  int64 upload_timestamp = 5;
  int64 category_level_one = 6;
  int64 request_timestamp = 7; 
}

message UserTimestamp {
  int64 request_id = 1;          
  int64 user_id = 2;             
  int64 request_timestamp = 3;    
  int64 device_id = 4;           
  int32 age = 5;                  
  int64 gender = 6;              
  int64 province = 7;
  repeated Item seq_effective_50 = 8;
}
```

Then, generate Python code from the .proto file using protoc:
```bash
# create proto
protoc --python_out=. ./inference/feature_insert/protos/recflow.proto
```

### Inserting Features into Redis Database
When storing user-side or item-side features in a Redis database, the process typically involves several steps:

​	1.	Create a message object.

​	2.	Assign values to each field of the message object.

​	3.	Serialize the message object.

​	4.	Store the serialized message object in the Redis database. The key is usually set as {dataset_name}:{object_name}:{object_primary_key}.

An example of inserting features into the Redis database using recflow is shown below:

```python
# connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Item
test_video_info = pd.read_feather('./inference/feature_data/recflow/realshow_test_video_info.feather')
for row in tqdm(test_video_info.itertuples(), total=len(test_video_info)):

    # 0. Create a message object
    item = recflow_pb2.Item()
    item.video_id = getattr(row, 'video_id')
    item.author_id = getattr(row, 'author_id')
    item.category_level_two = getattr(row, '_3')
    item.upload_type = getattr(row, 'upload_type')
    item.upload_timestamp = getattr(row, 'upload_timestamp')
    item.category_level_one = getattr(row, 'category_level_one')
    
    # 1. Serialize the Protobuf object into binary data
    serialized_data = item.SerializeToString()

    # 2. Store the compressed data in Redis
    r.set(f"recflow:item:{item.video_id}", serialized_data)
    

print("Item features are stored in Redis.")

# User
test_user_info = np.load('./inference/feature_data/recflow/test_user_info.npz')['arr_0']
for row in tqdm(test_user_info):

    # 0. Create a message object 
    user_timestamp = recflow_pb2.UserTimestamp()
    user_timestamp.request_id = row[0]
    user_timestamp.user_id = row[1]
    user_timestamp.request_timestamp = row[2]
    user_timestamp.device_id = row[3]
    user_timestamp.age = row[4]
    user_timestamp.gender = row[5]
    user_timestamp.province = row[6]
    
    for behavior in np.split(test_user_info[0][7:], len(test_user_info[0][7:]) // 6):
        item = user_timestamp.seq_effective_50.add()
        item.video_id = behavior[0]
        item.author_id = behavior[1]
        item.category_level_two = behavior[2]
        item.category_level_one = behavior[3]
        item.upload_type = behavior[4]
        item.request_timestamp = behavior[5]

    # 1. Serialize the Protobuf object into binary data
    serialized_data = user_timestamp.SerializeToString()

    # 2. Store the compressed data in Redis
    r.set(f"recflow:user_timestamp:{row[1]}_{row[2]}", serialized_data)

print("UserTimestamp features are stored in Redis.")
```

### Generate cache configuration file `feature_cache_config.yaml`

To enable the use of features stored in the cache, we need to generate a configuration file `feature_cache_config.yaml` for each dataset.

Taking Recflow as an example:

The `host`, `port`, and `db` fields specify details of Redis database. `features`  specifies the storage details for each feature. Within `features`, `key_temp` represents the key template for the feature in Redis database, where the content inside {} is replaced with specific item or user information, and `field` specifies the attribute name of the feature in the message object. `key_temp2proto` maps each key template to the corresponding message class name, which is used to create message objects.

```yaml
host: localhost
port: 6379
db: 0
features:
  video_id:
    key_temp: recflow:item:{video_id}
    field: video_id
  author_id:
    key_temp: recflow:item:{video_id}
    field: author_id
  category_level_two:
    key_temp: recflow:item:{video_id}
    field: category_level_two
...
  request_id:
    key_temp: recflow:user_timestamp:{user_id}_{request_timestamp}
    field: request_id
  user_id:
    key_temp: recflow:user_timestamp:{user_id}_{request_timestamp}
    field: user_id
  request_timestamp:
    key_temp: recflow:user_timestamp:{user_id}_{request_timestamp}
    field: request_timestamp
...
  seq_effective_50:
    key_temp: recflow:user_timestamp:{user_id}_{request_timestamp}
    field: seq_effective_50
key_temp2proto:
  recflow:item:{video_id}:
    class_name: Item
    module_path: ./inference/feature_insert/protos/recflow_pb2.py
  recflow:user_timestamp:{user_id}_{request_timestamp}:
    class_name: UserTimestamp
    module_path: ./inference/feature_insert/protos/recflow_pb2.py
```

Running ./inference/feature_insert/recflow_script/run.sh completes the three steps mentioned above.

## Inference

### InferenceEngine

[InferenceEngine](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py) class can be initialized to perform the inference process, which primarily consists of the following steps:

1. Converting a checkpoint of the recommendation model to an `onnxruntime.InferenceSession`.
2. Performing batch inference.
3. Outputting the top-k candidate items.

Here is an example demonstrating how to use InferenceEngine for batch inference in the ranking stage of Recflow dataset:

```python
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--infer_config_path", type=str, required=True, help="Inference config file")  
args = parser.parse_args()

with open(args.infer_config_path, 'r') as f:
    config = yaml.safe_load(f)

rank_inference_engine = RankerInferenceEngine(config)
infer_df = pd.read_feather('inference/inference_data/recflow/recflow_infer_data.feather')
item_df = pd.read_feather('inference/feature_data/recflow/realshow_test_video_info.feather')
all_item_ids = np.array(item_df['video_id'])
for batch_idx in range(10):
    print(f"This is batch {batch_idx}")
    batch_st = batch_idx * 128 
    batch_ed = (batch_idx + 1) * 128 
    batch_infer_df = infer_df.iloc[batch_st:batch_ed]
    batch_candidates = np.random.choice(all_item_ids, size=(128, 50))
    batch_candidates_df = pd.DataFrame({rank_inference_engine.feature_config['fiid']: batch_candidates.tolist()})
    retriever_outputs = rank_inference_engine.batch_inference(batch_infer_df, batch_candidates_df)
    print(type(retriever_outputs), retriever_outputs.shape)
```


### Converting a checkpoint to an InferenceSession
The [convert_to_onnx()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py#L63) function in InferenceEngine is used to convert the recommendation model’s checkpoint into an ONNX format. Based on the ONNX model, operator optimization can be considered using TensorRT. The ONNX model is further converted into an ONNX Runtime session or a TensorRT session for inference. Refer to the [convert_to_onnx()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/recflow_script/rank_stage.py#L24) function in the ranking stage of Recflow for more details.

### Batch inference

Batch inference is implemented in the [InferenceEngine.batch_inference()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py#L66) function. It includes the following steps:

​	1.	**Fetch user and context features**: The [get_user_context_features()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py#L98) function is used to obtain user and context features.

​	2.	**Fetch candidate item features**: The [get_candidates_features()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py#L148) function is used to obtain features for the candidate items.

​	3.	**Feed the features to the inference session**: Feed the features for inference to the inferenece session and return the top-k results.

### Customization

Users can extend the `InferenceEngine` class and override its methods to implement custom functionality:

​	•	[convert_to_onnx()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py#L63): Override this method to modify how to convert torch checkpoint to ONNX model.

​	•	[batch_inference()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py#L66): Override this method to customize the batch inference process.

​	•	[get_user_context_features()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py#L98): Override this method to customize how user and context features are fetched.

​	•	[get_candidates_features()](https://github.com/USTCLLM/RecStudio-Industry/blob/main/inference/inference/inference_engine.py#L148): Override this method to customize how candidate item features are fetched.

### Configuration

To initialize the `InferenceEngine` class, an inference configuration file `infer_config.yaml` is required. Using the Recflow dataset as an example, the meanings of each parameter in the configuration file are as follows:

​	•	`stage`: Specifies the stage of the inference process.

​	•	`model_ckpt_path`: Specifies the path to the recommendation model checkpoint.

​	•	`feature_cache_config_path`: Specifies the path to the feature cache configuration.

​	•	`output_topk`: Specifies the number of top-k items to return from the candidate set.

​	•	`infer_batch_size`: Specifies the batch size for inference.

​	•	`infer_device`: Specifies the GPU index to be used by the inference model.

```yaml
stage: rank 
model_ckpt_path: saved_model_demo/ranker_best_ckpt
feature_cache_config_path: inference/feature_insert/feature_cache_configs/recflow_feature_cache_config.yaml
output_topk: 10
infer_batch_size: 128
infer_device: 0
```

### Inference Service demo 
[inference_service.py](https://github.com/USTCLLM/RecStudio-Industry/blob/main/serve/inference_service.py) is a Gradio-based demo showcasing a multi-stage recommendation system inference pipeline using a retrieval and a ranker InferenceEngine. Refer to it for guidance on how to connect different stages of the InferenceEngine.
