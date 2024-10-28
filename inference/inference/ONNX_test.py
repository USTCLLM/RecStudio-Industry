import torch 
import onnx
import onnxruntime as ort
import numpy as np
from collections import OrderedDict

# 以tensor 为单位命名input_names，按照tensor出现的顺序依次命名
# 字典里tensor的顺序要和input_names里的对齐

# class SumModule(torch.nn.Module):
#     def forward(self, dict1, dict2):
#         y = dict2['b']
#         x = dict1['x']
#         # y = input['y']
#         return torch.sum(x, dim=1) + torch.prod(y, dim=-1)

# model = SumModule()
# model.eval()

# torch.onnx.export(
#     model,
#     # torch.ones(2, 2),
#     ({'x' : torch.ones(2, 2), 'y' : 2 * torch.ones(2, 2), 'z' : 2 * torch.ones(2, 2)},
#      {'a' : torch.ones(2, 2), 'b' : 2 * torch.ones(2, 2)},
#      {}), # 不知道为什么，结尾一定要有个字典。。。
#     # ({'x' : torch.ones(2, 2), 'y' : 2 * torch.ones(2, 2)},),
#     "onnx.pb",
#     input_names=["input1", "input2", "input3", "input4", "input5"],
#     output_names=["sum"],
#     opset_version=15,
#     verbose=True
# )



# # 加载模型
# onnx_model = onnx.load("onnx.pb")

# # 检查模型结构是否有效
# onnx.checker.check_model(onnx_model)

# # 访问模型中的节点信息（可选）
# print(onnx.helper.printable_graph(onnx_model.graph))



ort_session = ort.InferenceSession("onnx.pb")

outputs = ort_session.run(
    None,
    # {"input": {'x' : np.ones((2, 2)), 'y' : 2 * np.ones((2, 2))},
    #  "input2": {'a' : np.ones((2, 2)), 'b' : 2 * np.ones((2, 2))}}
    {
        "input1": np.ones((3, 2)).astype(np.float32),
        "input5": 3 * np.ones((3, 2)).astype(np.float32),
    }
)
print(outputs[0])