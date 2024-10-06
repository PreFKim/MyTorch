from .module import Module
from ..parameter import Param
import numpy as np
from ..ops.manipulate import stack

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
        if len(x.shape) == 2:
            _, l = x.shape
            out_shape = (self.out_channels, int((l+2*self.padding-self.kernel_size)/self.stride)+1)
        elif len(x.shape) == 3:
            b, _, l = x.shape
            out_shape = (b, self.out_channels, int((l+2*self.padding-self.kernel_size)/self.stride)+1)
        else:
            raise "Inputs shape should be 2D(Channels, Length) or 3D(Batch_size, Channels, Length)"
        
        ret = []
        for c in range(self.out_channels):
            for i in range(0, l-self.kernel_size+1, self.stride):
                ret.append((x[..., i:i+self.kernel_size] * self.weight[c]).sum((-1, -2)) ) # (((b,) in_channels, kernel_size)* (in_channels, kernel_size)).sum(-1, -2) -> (b,)
        
        ret = stack(ret, -1).reshape(out_shape)

        if self.bias is not None : return ret + self.bias
        else : return ret