import torch 
import numpy as np
import onnxruntime as ort

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict(context_input:dict, candidates_input:dict, output_topk:int):
    # context_input: dict 
    # candidates_input: dict
    # output_topk: int
    user_id = context_input['user_id'] % 10000 # [B]
    request_timestamp = context_input['request_timestamp'] * 2 % 10000 # [B]
    device_id = context_input['device_id'] * 3 % 10000 # [B]
    age = context_input['age'] * 4 % 10000 # [B]
    gender = context_input['gender'] * 5 % 10000 # [B]
    seq_effective_50 = context_input['seq_effective_50'] * 6 % 10000 # [B, 6 * L]
    
    video_id = candidates_input['candidates_video_id'] * 7 % 10000 # [B, N]
    author_id = candidates_input['candidates_author_id'] * 8 % 10000 # [B, N]
    category_level_two = candidates_input['candidates_category_level_two'] * 9 % 10000 # [B, N]
    upload_type = candidates_input['candidates_upload_type'] * 10 % 10000 # [B, N]
    upload_timestamp = candidates_input['candidates_upload_timestamp'] * 11 % 10000 # [B, N]
    category_level_one = candidates_input['candidates_category_level_one'] * 12 % 10000 # [B, N]

    range_t = torch.arange(100)
    
    return range_t.topk(output_topk).values.sum() + torch.sum(user_id) \
        + torch.sum(request_timestamp) \
        + torch.sum(device_id) \
        + torch.sum(age) \
        + torch.sum(gender) \
        + torch.sum(seq_effective_50) \
        + torch.sum(video_id) \
        + torch.sum(author_id) \
        + torch.sum(category_level_two) \
        + torch.sum(upload_type) \
        + torch.sum(upload_timestamp) \
        + torch.sum(category_level_one) 


user_id = torch.randint(100000, (5,))
request_timestamp = torch.randint(100000, (5,))
device_id = torch.randint(100000, (5,))
age = torch.randint(100000, (5,))
gender = torch.randint(100000, (5,))
seq_effective_50 = torch.randint(100000, (5, 300))
candidates_video_id = torch.randint(10000, (5, 16))
candidates_author_id = torch.randint(10000, (5, 16))
candidates_category_level_two = torch.randint(10000, (5, 16))
candidates_upload_type = torch.randint(10000, (5, 16))
candidates_upload_timestamp = torch.randint(10000, (5, 16))
candidates_category_level_one = torch.randint(10000, (5, 16))
output_topk = 6


context_input = {
    "user_id": user_id,
    "request_timestamp": request_timestamp,
    "device_id": device_id,
    "age": age,
    "gender": gender,
    "seq_effective_50": seq_effective_50,
}

candidates_input = {
    "candidates_video_id": candidates_video_id,  # [B, N]
    "candidates_author_id": candidates_author_id,  # [B, N]
    "candidates_category_level_two": candidates_category_level_two,  # [B, N]
    "candidates_upload_type": candidates_upload_type,  # [B, N]
    "candidates_upload_timestamp": candidates_upload_timestamp,  # [B, N]
    "candidates_category_level_one": candidates_category_level_one,  # [B, N]
}

print(f"predict answer is : {predict(context_input, candidates_input, output_topk)}")


ort_session = ort.InferenceSession("rank_model_onnx_demo.pb")

outputs = ort_session.run(
    None,
    {
        "user_id": to_numpy(user_id),
        "request_timestamp": to_numpy(request_timestamp),
        "device_id": to_numpy(device_id),
        "age": to_numpy(age),
        "gender": to_numpy(gender),
        "seq_effective_50": to_numpy(seq_effective_50),
        "candidates_video_id": to_numpy(candidates_video_id),
        "candidates_author_id": to_numpy(candidates_author_id),
        "candidates_category_level_two": to_numpy(candidates_category_level_two),
        "candidates_upload_type": to_numpy(candidates_upload_type),
        "candidates_upload_timestamp": to_numpy(candidates_upload_timestamp),
        "candidates_category_level_one": to_numpy(candidates_category_level_one),
        "output_topk": np.array(output_topk)
    }
)
print(f"ONNX model's answer: {outputs[0]}")