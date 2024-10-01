from .module import Module
from ..parameter import Param
import numpy as np

class Linear(Module):
    def __init__(self, in_channels, out_channels, bias=True):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Param(
            # np.ones((in_channels, out_channels)),
            np.random.uniform(-np.sqrt(6/in_channels), np.sqrt(6/in_channels), size=(in_channels, out_channels)), 
            requires_grad=True
            )

        self.bias = Param(
            # np.ones((out_channels)),
            np.random.uniform(-np.sqrt(1/in_channels), np.sqrt(1/in_channels), size=(out_channels)),
            requires_grad=True
            ) if bias else None
        
    def forward(self, x):
        if self.bias is not None : return x @ self.weight + self.bias
        else : return x @ self.weight 