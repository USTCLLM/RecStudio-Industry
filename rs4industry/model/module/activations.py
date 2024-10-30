import torch

__all__ = ['Dice', 'get_activation']

class Dice(torch.nn.Module):
    __constants__ = ['num_parameters']
    num_features: int

    def __init__(self, num_parameters, init: float = 0.0, epsilon: float = 1e-08):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = torch.nn.parameter.Parameter(
            torch.empty(1, num_parameters).fill_(init))
        self.epsilon = epsilon
        self.batch_norm = torch.nn.BatchNorm1d(num_parameters, eps=self.epsilon)

    def forward(self, x):
        x_shape = x.shape
        x_ = x.reshape(-1, x.shape[-1])
        x_normed = self.batch_norm(x_)
        p_x = torch.sigmoid(x_normed)
        p_x = p_x.reshape(x_shape)
        f_x = p_x * x + (1 - p_x) * x * self.weight.expand_as(x)
        return f_x

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)

def get_activation(activation: str, dim=None):
    if activation == None or isinstance(activation, torch.nn.Module):
        return activation
    elif type(activation) == str:
        if activation.lower() == 'relu':
            return torch.nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return torch.nn.Tanh()
        elif activation.lower() == 'leakyrelu':
            return torch.nn.LeakyReLU()
        elif activation.lower() == 'identity':
            return lambda x: x
        elif activation.lower() == 'dice':
            return Dice(dim)
        elif activation.lower() == 'gelu':
            return torch.nn.GELU()
        elif activation.lower() == 'leakyrelu':
            return torch.nn.LeakyReLU()
        else:
            raise ValueError(
                f'activation function type "{activation}"  is not supported, check spelling or pass in a instance of torch.nn.Module.')
    else:
        raise ValueError(
            '"activation_func" must be a str or a instance of torch.nn.Module. ')