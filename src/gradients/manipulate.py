from src.gradients.grad import Function, unbroadcast
import numpy as np

class Stack(Function):
    @staticmethod
    def forward(ctx, *inputs):
        *nodes, dim = inputs
        ctx.saved_for_backward(*nodes)
        ctx.dim = dim
        datas = [node.data for node in nodes]
        return np.stack(datas, dim)
    
    @staticmethod
    def backward(ctx, grad=1):
        nodes = ctx.saved_tensors
        dim = ctx.dim
        ret = [np.take(grad, indices=i, axis=dim) for i in range(len(nodes))]
        return ret

class Concat(Function):
    @staticmethod
    def forward(ctx, *inputs):
        *nodes, dim = inputs
        ctx.saved_for_backward(*nodes)
        ctx.dim = dim
        datas = [node.data for node in nodes]
        return np.concatenate(datas, dim)
    
    @staticmethod
    def backward(ctx, grad=1):
        nodes = ctx.saved_tensors
        dim = ctx.dim
        ret = [np.take(grad, indices=[i], axis=dim) for i in range(len(nodes))]
        return ret
    
class Reshape(Function):
    @staticmethod
    def forward(ctx, node_x, shape):
        ctx.saved_for_backward(node_x)
        return np.reshape(node_x.data, shape)
    
    @staticmethod
    def backward(ctx, grad=1):
        node_x, = ctx.saved_tensors
        return np.reshape(grad, node_x.shape)

class Max(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return np.maximum(node_x.data, node_y.data)
    
    @staticmethod
    def backward(ctx, grad=1):
        node_x, node_y = ctx.saved_tensors
        
        grad_x = np.zeros_like(grad)
        grad_y = np.zeros_like(grad)

        x_idx = node_x>=node_y
        y_idx = node_x<node_y
        grad_x[x_idx] = grad[x_idx]
        grad_y[y_idx] = grad[y_idx]

        grad_x = unbroadcast(grad_x, node_x.shape)
        grad_y = unbroadcast(grad_y, node_x.shape)
        return grad_x, grad_y

class Min(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return np.minimum(node_x.data, node_y.data)
    
    @staticmethod
    def backward(ctx, grad=1):
        node_x, node_y = ctx.saved_tensors
        
        grad_x = np.zeros_like(grad)
        grad_y = np.zeros_like(grad)

        x_idx = node_x<=node_y
        y_idx = node_x>node_y
        grad_x[x_idx] = grad[x_idx]
        grad_y[y_idx] = grad[y_idx]

        grad_x = unbroadcast(grad_x, node_x.shape)
        grad_y = unbroadcast(grad_y, node_x.shape)
        return grad_x, grad_y