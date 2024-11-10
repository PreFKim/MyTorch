from src.gradients.grad import Function, unbroadcast
import numpy as np


class Add(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return node_x.data + node_y.data

    @staticmethod
    def backward(ctx, grad=1):
        node_x, node_y = ctx.saved_tensors
        grad_x = unbroadcast(grad, node_x.shape)
        grad_y = unbroadcast(grad, node_y.shape)
        return grad_x, grad_y

class Sub(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return node_x.data - node_y.data

    @staticmethod
    def backward(ctx, grad=1):
        node_x, node_y = ctx.saved_tensors
        grad_x = unbroadcast(grad, node_x.shape)
        grad_y = unbroadcast(grad, node_y.shape)
        return grad_x, -grad_y


class Mul(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return node_x.data * node_y.data

    @staticmethod
    def backward(ctx, grad=1):
        node_x, node_y = ctx.saved_tensors
        grad_x = unbroadcast(grad * node_y.data, node_x.shape)
        grad_y = unbroadcast(grad * node_x.data, node_y.shape)
        return grad_x, grad_y

class Div(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return node_x.data / node_y.data

    @staticmethod
    def backward(ctx, grad=1):
        node_x, node_y = ctx.saved_tensors
        grad_x = unbroadcast(grad * (1 / node_y.data), node_x.shape)
        grad_y = unbroadcast(grad * -(node_x.data / (node_y.data)**2), node_y.shape)
        return grad_x, grad_y

class FloorDiv(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return node_x.data // node_y.data

class Mod(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return node_x.data % node_y.data

class Pow(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):
        ctx.saved_for_backward(node_x, node_y)
        return node_x.data ** node_y.data

    @staticmethod
    def backward(ctx, grad=1):
        node_x, node_y = ctx.saved_tensors
        grad_x = unbroadcast(grad * (node_y.data * node_x.data ** (node_y.data - 1)), node_x.shape)
        grad_y = unbroadcast(grad * (node_x.data ** node_y.data * np.log(node_x.data)), node_y.shape)
        return grad_x, grad_y

class Abs(Function):
    @staticmethod
    def forward(ctx, node_x):
        ctx.saved_for_backward(node_x)
        return np.abs(node_x.data)

    @staticmethod
    def backward(ctx, grad=1):
        node, = ctx.saved_tensors
        ret = grad.copy()
        ret[node.data<0] = -ret[node.data<0]
        return ret
    
class Neg(Function):
    @staticmethod
    def forward(ctx, node_x):
        ctx.saved_for_backward(node_x)
        return -node_x.data

    @staticmethod
    def backward(ctx, grad=1):
        return -grad

class MatMul(Function):
    @staticmethod
    def forward(ctx, node_x, node_y):  
        ctx.saved_for_backward(node_x, node_y)
        return node_x.data @ node_y.data  

    @staticmethod
    def backward(ctx, grad=1):
        node_x, node_y = ctx.saved_tensors
        grad_x = unbroadcast(grad @ node_y.data.transpose(-1, -2), node_x.shape)
        grad_y = unbroadcast(node_x.data.transpose(-1, -2) @ grad, node_y.shape)
        return grad_x, grad_y

class Log(Function):
    @staticmethod
    def forward(ctx, node_x):
        ctx.saved_for_backward(node_x)
        return np.log(np.clip(node_x.data, 1e-8, np.inf))

    @staticmethod
    def backward(ctx, grad=1):
        node_x, = ctx.saved_tensors
        return grad / node_x.data

class Sum(Function):
    @staticmethod
    def forward(ctx, node_x, dim, keepdim):  
        ctx.saved_for_backward(node_x)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return node_x.data.sum(axis=dim, keepdims=keepdim)

    @staticmethod
    def backward(ctx, grad=1):
        node_x, = ctx.saved_tensors
        dim = ctx.dim
        keepdim = ctx.keepdim

        input_shape = node_x.shape

        if dim is None:
            if not keepdim:
                grad = grad.reshape(1)
            return np.broadcast_to(grad, input_shape)

        if isinstance(dim, (int, np.integer)):
            dim = [dim]
        elif isinstance(dim, tuple):
            dim = list(dim)

        for i in range(len(dim)):
            if dim[i] < 0:
                dim[i] = len(input_shape) + dim[i]
        dim = sorted(dim)

        if not keepdim:
            output_shape = [size for i, size in enumerate(input_shape) if i not in dim]
            grad = grad.reshape(output_shape)

            for d in dim:
                grad = np.expand_dims(grad, d)

        return np.broadcast_to(grad, input_shape)
    
class Mean(Function):
    @staticmethod
    def forward(ctx, node_x, dim, keepdim):  
        ctx.saved_for_backward(node_x)
        ctx.dim = dim
        ctx.keepdim = keepdim
        return node_x.data.mean(axis=dim, keepdims=keepdim)

    @staticmethod
    def backward(ctx, grad=1):
        node_x, = ctx.saved_tensors
        dim = ctx.dim
        keepdim = ctx.keepdim

        input_shape = node_x.shape

        if dim is None:
            div = np.prod(input_shape)
            if not keepdim:
                grad = grad.reshape(1)
            return np.broadcast_to(grad, input_shape) / div

        if isinstance(dim, (int, np.integer)):
            dim = [dim]
        elif isinstance(dim, tuple):
            dim = list(dim)

        for i in range(len(dim)):
            if dim[i] < 0:
                dim[i] = len(input_shape) + dim[i]
        dim = sorted(dim)
        
        div = np.prod([input_shape[i] for i in dim])

        if not keepdim:
            # Calculate output shape
            output_shape = [size for i, size in enumerate(input_shape) if i not in dim]
            grad = grad.reshape(output_shape)

        if not keepdim:
            for d in dim:
                grad = np.expand_dims(grad, d)

        return np.broadcast_to(grad, input_shape) / div
    