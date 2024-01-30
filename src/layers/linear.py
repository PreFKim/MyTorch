from ..array import from_numpy
from .module import Module
import numpy as np

class Linear(Module):
    def __init__(self,in_channels,out_channels,bias=True):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.w = from_numpy(np.random.uniform(-np.sqrt(6/in_channels),np.sqrt(6/in_channels),size=(in_channels,out_channels)))

        if self.bias:
            self.b = from_numpy(np.random.uniform(-np.sqrt(1/in_channels),np.sqrt(1/in_channels),size=(out_channels)))
        
        
    def forward(self,x):
        if self.bias : return x @ self.w + self.b
        else : return x @ self.w 