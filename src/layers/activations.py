
from src.layers.module import Module
from src.gradients.manipulate import Max
from src.parameter import operation
import numpy as np
from src.ops.basic import exp

class ReLU(Module):
    def forward(self, x):
        return operation(Max, x, 0)

class Sigmoid(Module):
    def forward(self, x):
        return 1/(1+exp(-x))

class Tanh(Module):
    def forward(self, x):
        e_x = exp(x)
        neg_e_x = exp(-x)
        return (e_x-neg_e_x)/(e_x+neg_e_x)

class SiLU(Module):
    def forward(self, x):
        return x*(1/(1+exp(-x)))
    
class Softmax(Module):
    def forward(self, x):
        e_x = exp(x-np.max(x.data, -1, keepdims=True))
        return e_x / e_x.sum(-1, keepdim=True)