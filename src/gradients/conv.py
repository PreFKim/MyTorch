from .grad import Grad
import numpy as np

class Convolution(Grad):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(node, weight, stride, padding, bias):
        
        out_channels = weight.shape[0]
        kernel_size = weight.shape[2]
        out_length = int((node.shape[-1]+2*padding-kernel_size)/stride)+1
        if len(node.shape) == 2:
            c, l = node.shape
            out_shape = (out_channels, out_length)
            pad_width = ((0, 0), (padding, padding))
        elif len(node.shape) == 3:
            b, c, l = node.shape
            out_shape = (b, out_channels, out_length)
            pad_width = ((0, 0), (0, 0), (padding, padding))
        else:
            raise "Inputs shape should be 2D(Channels, Length) or 3D(Batch_size, Channels, Length)"

        node_data = np.pad(node.data, pad_width=pad_width)

        left = (kernel_size-1)//2
        right = kernel_size//2

        ret = np.zeros(out_shape)
        for c in range(out_channels):
            for i in range(left, l+2*padding-right, stride):
                ret[..., c, (i-left)//stride] = (node_data[..., i-left:i+right+1] * weight.data[c]).sum((-1, -2)) # batch, in_channels, kernel_size -> batch, 1, 1
        
        if bias is not None:
            ret += bias.data
        return ret
    
    def backward(self, grad=1):
        # grad shape: out_shape((b) c, l)
        node, weight, stride, padding, bias = self.saved_tensors

        kernel_size = weight.shape[2]        
        if len(node.shape) == 2:
            c, l = node.shape
            grad_node = np.zeros((c, l+padding*2))
            grad_weight = np.zeros_like((weight.data)) 
            pad_width = ((0, 0), (padding, padding))
        elif len(node.shape) == 3:
            b, c, l = node.shape
            grad_node = np.zeros((b, c, l+padding*2))
            grad_weight = np.zeros((b, *weight.shape)) 
            pad_width = ((0, 0), (0, 0), (padding, padding))
        
        node_data = np.pad(node.data, pad_width=pad_width)

        left = (kernel_size-1)//2
        right = kernel_size//2
        for i in range(grad.shape[-1]):
            grad_node[..., (i*stride): (i*stride)+left+right+1] += (weight.data * grad[..., np.newaxis, i:i+1]).sum(-3)
            # grad_node -> (batch), in_channels, out_length+2*padding
            # weight -> out_channels, in_channels, kernel_size 
            # grad -> (batch), out_chennels, out_length -> indexing -> (batch), out_channels, 1, 1
            # weight * grad -> (batch), out_channels, in_channels, kernel_size -> (batch), in_channels, kernel_size
        grad_node = grad_node[..., padding:padding+l]

        for i in range(grad.shape[-1]):
            grad_weight += node_data[..., np.newaxis, :, (i*stride):(i*stride)+left+right+1] * grad[..., np.newaxis, i:i+1]
            # node -> (batch), in_channels, in_length -> indexing -> (batch), 1, in_channels, kernel_size
            # grad -> (batch), out_chennels, out_length -> indexing -> (batch), out_chennels, 1, 1
            # node * grad -> (batch), out_channels, in_channels, kernel_size -> (batch), out_channels, kernel_size

        return grad_node, grad_weight, grad if bias is not None else None