
from .module import Module
from ..parameter import Param
import numpy as np

class ReLU(Module):
        
    def forward(self, x):
        x[x<0] = 0
        return x
    