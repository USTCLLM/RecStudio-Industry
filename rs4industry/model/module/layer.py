from typing import Callable
import torch

from rs4industry.model.module import get_activation

class MLPModule(torch.nn.Module):
    def __init__(
            self, 
            mlp_layers, 
            activation_func='ReLU', 
            dropout=0.0, 
            bias=True, 
            batch_norm=False,
            last_activation=True,
            last_bn=True
        ):
        """
        MLPModule
        Gets a MLP easily and quickly.

        Args:
            mlp_layers(list): the dimensions of every layer in the MLP.
            activation_func(torch.nn.Module,str,None): the activation function in each layer.
            dropout(float): the probability to be set in dropout module. Default: ``0.0``.
            bias(bool): whether to add batch normalization between layers. Default: ``False``.
            last_activation(bool): whether to add activation in the last layer. Default: ``True``.
            last_bn(bool): whether to add batch normalization in the last layer. Default: ``True``.

        Examples:
        >>> MLP = MLPModule([64, 64, 64], 'ReLU', 0.2)
        >>> MLP.model
        Sequential(
            (0): Dropout(p=0.2, inplace=False)
            (1): Linear(in_features=64, out_features=64, bias=True)
            (2): ReLU()
            (3): Dropout(p=0.2, inplace=False)
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): ReLU()
        )
        >>> MLP.add_modules(torch.nn.Linear(64, 10, True), torch.nn.ReLU())
        >>> MLP.model
        Sequential(
            (0): Dropout(p=0.2, inplace=False)
            (1): Linear(in_features=64, out_features=64, bias=True)
            (2): ReLU()
            (3): Dropout(p=0.2, inplace=False)
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): ReLU()
            (6): Linear(in_features=64, out_features=10, bias=True)
            (7): ReLU()
        )
        """
        super().__init__()
        self.mlp_layers = mlp_layers
        self.batch_norm = batch_norm
        self.bias = bias
        self.dropout = dropout
        self.activation_func = activation_func
        self.model = []
        last_bn = self.batch_norm and last_bn
        for idx, layer in enumerate((zip(self.mlp_layers[: -1], self.mlp_layers[1:]))):
            self.model.append(torch.nn.Dropout(dropout))
            self.model.append(torch.nn.Linear(*layer, bias=bias))
            if (idx == len(mlp_layers)-2 and last_bn) or (idx < len(mlp_layers)-2 and batch_norm):
                self.model.append(torch.nn.BatchNorm1d(layer[-1]))
            if ( (idx == len(mlp_layers)-2 and last_activation and activation_func is not None)
                or (idx < len(mlp_layers)-2 and activation_func is not None) ):
                activation = get_activation(activation_func, dim=layer[-1])
                self.model.append(activation)
        self.model = torch.nn.Sequential(*self.model)

    def add_modules(self, *args):
        """
        Adds modules into the MLP model after obtaining the instance.

        Args:
            args(variadic argument): the modules to be added into MLP model.
        """
        for block in args:
            assert isinstance(block, torch.nn.Module)

        for block in args:
            self.model.add_module(str(len(self.model._modules)), block)

    def forward(self, x):
        shape = x.shape
        output = self.model(x.view(-1, shape[-1]))
        return output.view(*shape[:-1], -1)

    # def __repr__(self):
    #     return (f"mlp_layers={self.mlp_layers}, activation_func={self.activation_func},\n"
    #            f"\tdropout={self.dropout},\nbatch_norm={self.batch_norm},\n"
    #            f"\tlast_activation={self.last_activation}\n"
    #            f"\tbias={self.bias}")


class LambdaModule(torch.nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)