import torch
from torch import nn 

class MyRankModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(10000, 64)
        self.item_embedding = nn.Embedding(10000, 64)
        self.linear_0 = nn.Linear(128, 64) 
        self.linear_1 = nn.Linear(64, 1) 
    
    def forward(self):
        pass 

    # def predict(self, context_input:dict, candidates_input:dict, output_topk:int):
    #     # context_input: dict 
    #     # candidates_input: dict
    #     # output_topk: int
    #     user_id = context_input['user_id'] % 10000 # [B]
    #     request_timestamp = context_input['request_timestamp'] % 10000 # [B]
    #     device_id = context_input['device_id'] % 10000 # [B]
    #     age = context_input['age'] % 10000 # [B]
    #     gender = context_input['gender'] % 10000 # [B]
    #     seq_effective_50 = context_input['seq_effective_50'] % 10000 # [B, 6 * L]

    #     video_id = candidates_input['video_id'] # [B, N]

    #     input_embed = torch.concat(
    #         [self.user_embedding(user_id).unsqueeze(-2), self.user_embedding(video_id)],
    #         dim=-1
    #     ) # [B, D], [B, N, D] -> [B, N, 2D]
    #     scores = self.linear_1(self.linear_0(input_embed)).squeeze() # [B, N]
    #     topk_idx = torch.topk(scores, dim=-1).indices # [B, topk]
    #     return torch.gather(video_id, dim=-1, index=topk_idx)

    @torch.no_grad()
    def predict(self, context_input:dict, candidates_input:dict, output_topk:int):
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
            
    def eval(self):
        self.forward = self.predict
        return super().eval()
    

model = MyRankModel()
model.eval()


context_input = {
    "user_id": torch.randint(100000, (5,)),
    "request_timestamp": torch.randint(100000, (5,)),
    "device_id": torch.randint(100000, (5,)),
    "age": torch.randint(100000, (5,)),
    "gender": torch.randint(100000, (5,)),
    "seq_effective_50": torch.randint(100000, (5, 300)),
}
candidates_input = {
    "candidates_video_id": torch.randint(10000, (5, 16)),  # [B, N]
    "candidates_author_id": torch.randint(10000, (5, 16)),  # [B, N]
    "candidates_category_level_two": torch.randint(10000, (5, 16)),  # [B, N]
    "candidates_upload_type": torch.randint(10000, (5, 16)),  # [B, N]
    "candidates_upload_timestamp": torch.randint(10000, (5, 16)),  # [B, N]
    "candidates_category_level_one": torch.randint(10000, (5, 16)),  # [B, N]
}
output_topk = 6
torch.onnx.export(
    model,
    (context_input, candidates_input, output_topk), 
    "rank_model_onnx_demo.pb",
    input_names=['user_id', 'request_timestamp', 'device_id', 'age', 'gender', 'seq_effective_50', 
                'candidates_video_id', 'candidates_author_id', 'candidates_category_level_two', 'candidates_upload_type', 
                'candidates_upload_timestamp', 'candidates_category_level_one', 'output_topk'],
    output_names=["test_output"],
    dynamic_axes={
        "user_id": {0: "batch_size"},
        "request_timestamp": {0: "batch_size"},
        "device_id": {0: "batch_size"},
        "age": {0: "batch_size"},
        "gender": {0: "batch_size"},
        "seq_effective_50": {0: "batch_size"},
        "candidates_video_id": {0: "batch_size", 1: "num_candidates"},  # [B, N]
        "candidates_author_id": {0: "batch_size", 1: "num_candidates"},  # [B, N]
        "candidates_category_level_two": {0: "batch_size", 1: "num_candidates"},  # [B, N]
        "candidates_upload_type": {0: "batch_size", 1: "num_candidates"},  # [B, N]
        "candidates_upload_timestamp": {0: "batch_size", 1: "num_candidates"},  # [B, N]
        "candidates_category_level_one": {0: "batch_size", 1: "num_candidates"},  # [B, N]
    },
    opset_version=15,
    verbose=True
)
print('model saved!')

         



