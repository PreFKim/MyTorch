from ..parameter import Param, operation
from ..gradients.manipulate import Stack, Concat
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
    return Param(np.zeros(shape), requires_grad=requires_grad)

def zeros_like(node, requires_grad=False):
    return Param(np.zeros_like(node), requires_grad=requires_grad)

def ones(shape, requires_grad=False):
    return Param(np.ones(shape), requires_grad=requires_grad)

def ones_like(node, requires_grad=False):
    return Param(np.ones_like(node), requires_grad=requires_grad)

def full(shape, value, requires_grad=False):
    return Param(np.full(shape, value), requires_grad=requires_grad)

def full_like(node, requires_grad=False):
    return Param(np.full_like(node), requires_grad=requires_grad)

def reshape(node, shape):
    return node.reshape(shape)

def arange(*inputs, requires_grad=True):
    return Param(np.arange(*inputs), requires_grad=requires_grad)