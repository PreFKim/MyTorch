from src.layers.module import Module
from src.parameter import Param, operation
import numpy as np
from src.gradients.conv import Convolution

def _2tuple(value, n=1):
    if isinstance(value, int):
        return tuple(value for _ in range(n))
    elif isinstance(value, (tuple, list)):
        if len(value) != n:
            raise ValueError("len(value) != n")
        return tuple(value)
    else: 
        raise ValueError("Value type should be int, tuple, list")

class Convnd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, n=1):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _2tuple(kernel_size, n)
        self.stride = _2tuple(stride, n)
        self.padding = _2tuple(padding, n)
        
        # kaiming uniform init
        # fan = _calculate_correct_fan(tensor, mode) # in_channels
        # gain = calculate_gain(nonlinearity, a) # math.sqrt(2.0 / (1 + a**2)) [a=5] -> 0.2773500981126146
        # std = gain / math.sqrt(fan) # gain / (in_channels) **(1/2)
        # bound = math.sqrt(3.0) * std # 1.7320508075688772 * std
        
        bound = 1.7320508075688772 * 0.2773500981126146 / (in_channels)**(1/2)

        self.weight = Param(
            # np.ones((out_channels, in_channels, *self.kernel_size)).astype(np.float32),
            # np.random.uniform(-np.sqrt(6/in_channels), np.sqrt(6/in_channels), size=(out_channels, in_channels, *self.kernel_size)).astype(np.float32), 
            # np.random.normal(loc=0, scale=np.sqrt(2/in_channels), size=(out_channels, in_channels, *self.kernel_size)).astype(np.float32),
            np.random.uniform(-bound, bound, size=(out_channels, in_channels, *self.kernel_size)).astype(np.float32), 
            requires_grad=True
            )
        self.bias = Param(
            # np.ones((out_channels, *[1 for _ in self.kernel_size])).astype(np.float32),
            np.random.uniform(-np.sqrt(1/in_channels), np.sqrt(1/in_channels), size=(out_channels, *[1 for _ in self.kernel_size])).astype(np.float32),
            requires_grad=True
            ) if bias else None

    def forward(self, x):
        return operation(Convolution, x, self.weight, self.stride, self.padding, self.bias, convert=False)
    
class Conv1d(Convnd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, 1)
        
class Conv2d(Convnd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, 2)