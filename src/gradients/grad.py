import numpy as np

def unbroadcast(grad, shape):
    if grad.shape != shape:
        # Un broadcasting
        grad = grad.sum(tuple(range(len(grad.shape)-len(shape))))
        grad = grad.sum(tuple(i for i in range(len(shape)) if shape[i] != grad.shape[i]), keepdims=True)
    return grad


class Function: # Role for torch.autograd.Function
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError("Backward operation for this operation isn't implemened")
    
    @staticmethod
    def backward(ctx, grad=1):
        raise NotImplementedError("Backward operation for this operation isn't implemened")

class Accumulate:
    def __init__(self, node):
        self.node = node
    
    def backward(self, grad=1):
        self.node.grad = self.node.grad + grad if self.node.grad is not None else grad

class ContextManager:
    def __init__(self):
        self.saved_tensors = []
        self.saved_versions = []

    def saved_for_backward(self, *nodes):
        self.saved_tensors = nodes
        self.saved_versions = [node._version for node in nodes]

class GradFunction:
    def __init__(self, op, ctx, next_functions):
        self.ctx = ctx
        self.op = op
        self.next_functions = next_functions

    def backward(self, grad=1):
        for init_version, node in zip(self.ctx.saved_versions, self.ctx.saved_tensors):
            if init_version != node._version:
                raise RuntimeError(
                        "One of the differentiated Node appears to have been "
                        "modified in-place since being used for gradient computation."
                    )
            
        calc_grads = self.op.backward(self.ctx, grad)
        if not isinstance(calc_grads, (tuple, list)):
            calc_grads = [calc_grads]

        if len(calc_grads) != len(self.next_functions):
            raise ValueError(f"Length mismatch: calc_grads({len(calc_grads)}) != next_functions({len(self.next_functions)})")
        
        for calc, next_fn in zip(calc_grads, self.next_functions):
            if next_fn is not None:
                next_fn.backward(calc)
