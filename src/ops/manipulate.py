from src.parameter import Param, operation
from src.gradients.manipulate import Stack, Concat, Max, Min
import numpy as np

def stack(nodes, dim=-1):
    requires_grad = False
    for node in nodes:
        requires_grad = requires_grad or node.requires_grad
    return operation(Stack, nodes, dim, convert=False, requires_grad=requires_grad)

def concat(nodes, dim=-1):
    requires_grad = False
    for node in nodes:
        requires_grad = requires_grad or node.requires_grad
    return operation(Concat, nodes, dim, convert=False, requires_grad=requires_grad)

def zeros(shape, requires_grad=False):
    return Param(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

def zeros_like(node, requires_grad=False):
    return Param(np.zeros_like(node.data), requires_grad=requires_grad)

def ones(shape, requires_grad=False):
    return Param(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

def ones_like(node, requires_grad=False):
    return Param(np.ones_like(node.data), requires_grad=requires_grad)

def full(shape, value, requires_grad=False):
    return Param(np.full(shape, value, dtype=np.float32), requires_grad=requires_grad)

def full_like(node, requires_grad=False):
    return Param(np.full_like(node.data), requires_grad=requires_grad)

def reshape(node, shape):
    return node.reshape(shape)

def arange(*inputs, requires_grad=True):
    return Param(np.arange(*inputs, dtype=np.float32), requires_grad=requires_grad)

def maximum(x, y):
    return operation(Max, x, y)

def minimum(x, y):
    return operation(Min, x, y)