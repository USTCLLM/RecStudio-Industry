import torch
class SumModule(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x['a'], dim=1)


torch.onnx.export(
    SumModule(),
    ({'a': torch.ones(2, 2)}, {}), # 当输入是字典的时候，元组后面需要加一个空字典
    "onnx.pb",
    input_names=["x"],
    output_names=["sum"],
)