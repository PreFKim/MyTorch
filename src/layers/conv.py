from .module import Module
from ..parameter import Param, operation
import numpy as np
from ..ops.manipulate import stack
from ..gradients import Convolution

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Param(
            np.ones((out_channels, in_channels, kernel_size)),
            # np.random.uniform(-np.sqrt(6/in_channels), np.sqrt(6/in_channels), size=(out_channels, in_channels, kernel_size)), 
            requires_grad=True
            )

        self.bias = Param(
            np.ones((out_channels, 1)),
            # np.random.uniform(-np.sqrt(1/in_channels), np.sqrt(1/in_channels), size=(out_channels, 1)),
            requires_grad=True
            ) if bias else None
        
    def forward(self, x):
        return operation(Convolution, x, self.weight, self.stride, self.padding, self.bias, convert=False)