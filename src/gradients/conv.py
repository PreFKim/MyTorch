from src.gradients.grad import Function, unbroadcast
import numpy as np

class Convolution(Function):
    @staticmethod
    def forward(ctx, node, weight, stride, padding, bias):
        # weight.shape -> out_channels, in_channels, ... (nd)
        ctx.saved_for_backward(node, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        kernel_size = weight.shape[2:]

        if len(node.shape) == len(weight.shape)-1 : 
            out_shape = [weight.shape[0]] + [int((node.shape[1 + i]+2*p-k)/s)+1 for i, (k, s, p) in enumerate(zip(kernel_size, stride, padding))]
            pad_width = [(0, 0)] +  [(p, p) for p in padding]
        elif len(node.shape) == len(weight.shape):
            out_shape = [node.shape[0], weight.shape[0]] + [int((node.shape[2+i]+2*p-k)/s)+1 for i, (k, s, p) in enumerate(zip(kernel_size, stride, padding))]
            pad_width = [(0, 0), (0, 0)] +  [(p, p) for p in padding]
        else :
            raise ValueError(f"Inputs shape should be {len(weight.shape)-1}D(Channels, ...) or {len(weight.shape)}D(Batch_size, Channels, ..)")
            
        node_data = np.pad(node.data, pad_width=pad_width)

        if len(weight.shape) == 3: # Conv1d
            ret = Conv1d_forward(node_data, weight, out_shape, stride)
        elif len(weight.shape) == 4: # Conv2d
            ret = Conv2d_forward(node_data, weight, out_shape, stride)
        else:
            raise NotImplementedError(f"Conv{len(weight.shape)-2}d operation is not implemented")

        if bias is not None:
            ret += bias.data
        return ret
    
    @staticmethod
    def backward(ctx, grad=1):
        # grad shape: out_shape((b) c, l)
        node, weight, bias = ctx.saved_tensors    
        stride = ctx.stride
        padding = ctx.padding

        if len(node.shape) == len(weight.shape) -1:
            batch_first = False
            pad_width = [(0, 0)] +  [(p, p) for p in padding]
        elif len(node.shape) == len(weight.shape):
            batch_first = True
            pad_width = [(0, 0), (0, 0)] +  [(p, p) for p in padding]
        
        node_data = np.pad(node.data, pad_width=pad_width)

        if len(weight.shape) == 3: # Conv1d
            grad_node, grad_weight = Conv1d_backward(node_data, weight, grad, stride, padding, batch_first)
        elif len(weight.shape) == 4: # Conv2d
            grad_node, grad_weight = Conv2d_backward(node_data, weight, grad, stride, padding, batch_first)
        else:
            raise NotImplementedError(f"Conv{len(weight.shape)-2}d operation is not implemented")

        grad_node = unbroadcast(grad_node, node.shape)
        grad_weight = unbroadcast(grad_weight, weight.shape)
        grad_bias = unbroadcast(grad, bias.shape)
        return grad_node, grad_weight, grad_bias if bias is not None else None

def Conv1d_forward(data, weight, out_shape, stride):
    out_channels, in_channels, *kernel_size = weight.shape
    l, = data.shape[-1:]

    left = (kernel_size[0]-1)//2
    right = kernel_size[0]//2

    ret = np.zeros(out_shape, dtype=data.dtype)
    for c in range(out_channels): 
        for i in range(left, l-right, stride[0]):
            ret[..., c, (i-left)//stride[0]] = (data[..., i-left:i+right+1] * weight.data[c]).sum((-1, -2)) # batch, in_channels, kernel_size -> batch, 1, 1
    return ret

def Conv1d_backward(data, weight, grad, stride, padding, batch_first):
    kernel_size = weight.shape[2:]
    l, = data.shape[-1:]
    grad_l, = grad.shape[-1:]

    left = (kernel_size[0]-1)//2
    right = kernel_size[0]//2

    grad_node = np.zeros_like(data)
    grad_weight = np.zeros((data.shape[0], *weight.shape), dtype=data.dtype) if batch_first else np.zeros_like(weight)

    for i in range(grad_l):
        grad_node[..., (i*stride[0]): (i*stride[0])+left+right+1] += (weight.data * grad[..., np.newaxis, i:i+1]).sum(-3)
        # grad_node -> (batch), in_channels, out_length+2*padding
        # weight -> out_channels, in_channels, kernel_size 
        # grad -> (batch), out_chennels, out_length -> indexing -> (batch), out_channels, 1, 1
        # weight * grad -> (batch), out_channels, in_channels, kernel_size -> (batch), in_channels, kernel_size
    
    grad_node = grad_node[..., padding[0]:l-padding[0]]

    for i in range(grad_l):
        grad_weight += data[..., np.newaxis, :, (i*stride[0]):(i*stride[0])+left+right+1] * grad[..., np.newaxis, i:i+1]
        # node -> (batch), in_channels, in_length -> indexing -> (batch), 1, in_channels, kernel_size
        # grad -> (batch), out_chennels, out_length -> indexing -> (batch), out_chennels, 1, 1
        # node * grad -> (batch), out_channels, in_channels, kernel_size -> (batch), out_channels, kernel_size
    return grad_node, grad_weight

def Conv2d_forward(data, weight, out_shape, stride):
    out_channels, in_channels, *kernel_size = weight.shape
    h, w = data.shape[-2:]

    top = (kernel_size[0]-1)//2
    bottom = kernel_size[0]//2 
    left = (kernel_size[1]-1)//2
    right = kernel_size[1]//2

    ret = np.zeros(out_shape, dtype=data.dtype)
    for c in range(out_channels): 
        for i in range(top, h-bottom, stride[0]):
            for j in range(left, w-right, stride[1]):
                ret[..., c, (i-top)//stride[0], (j-left)//stride[1]] = (data[..., i-top:i+bottom+1, j-left:j+right+1] * weight.data[c]).sum((-1, -2, -3)) 
    return ret

def Conv2d_backward(data, weight, grad, stride, padding, batch_first):
    kernel_size = weight.shape[2:]
    h, w = data.shape[-2:]
    grad_h, grad_w = grad.shape[-2:]

    top = (kernel_size[0]-1)//2
    bottom = kernel_size[0]//2 
    left = (kernel_size[1]-1)//2
    right = kernel_size[1]//2

    grad_node = np.zeros_like(data)
    grad_weight = np.zeros((data.shape[0], *weight.shape), dtype=data.dtype) if batch_first else np.zeros_like(weight)
    
    for i in range(grad_h):
        for j in range(grad_w):
            grad_node[..., (i*stride[0]): (i*stride[0])+top+bottom+1, (j*stride[1]): (j*stride[1])+left+right+1] += (weight.data * grad[..., np.newaxis, i:i+1, j:j+1]).sum(-4)
    
    grad_node = grad_node[..., padding[0]:h-padding[0], padding[1]:w-padding[1]]

    for i in range(grad_h):
        for j in range(grad_w):
            grad_weight += data[..., np.newaxis, :, (i*stride[0]):(i*stride[0])+top+bottom+1, (j*stride[1]):(j*stride[1])+left+right+1] * grad[..., np.newaxis, i:i+1, j:j+1]
        
    return grad_node, grad_weight
