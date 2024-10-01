import numpy as np
from .gradients.basic import *
from .gradients.index import Get, Set

class Param:
    def __init__(self, data=0.0, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.data = data 
        elif isinstance(data, (int, float, complex, list, tuple)) :
            self.data = np.array(data)
        else :
            raise "Not supported data type"
        
        self.grad = None
        self.grad_fn = None
        self.requires_grad = requires_grad
        self.is_leaf = True
        self.shape = self.data.shape
        
    def __repr__(self):
        return f"Data:{self.data}"+ (f", requrired_grad:{self.requires_grad}" if self.requires_grad else "")
    
    def __len__(self):
        return self.shape[0]
    
    def backward(self, grad=1):

        if self.requires_grad:
            if not isinstance(grad, np.ndarray) :
                grad = np.ones_like(self.data) * grad
            
            if grad.shape != self.shape:
                # Un broadcasting
                ln_grad = len(grad.shape)
                ln_self = len(self.shape)
                
                grad = grad.sum(tuple(range(ln_grad-ln_self)))
                
                dims = []
                for i in range(ln_self):
                    if self.shape[i]==1 and grad.shape[i]:
                        dims.append(i)
                grad = grad.sum(tuple(dims), keepdims=True)

            if self.grad_fn is not None:                
                calc_grads = self.grad_fn.backward(grad)

                if not isinstance(calc_grads, (tuple, list)):
                    calc_grads = [calc_grads]

                grad_idx = 0
                for i in range(len(self.grad_fn)):
                    prev_node = self.grad_fn.saved_tensors[i]
                    if isinstance(prev_node, Param):
                        prev_node.backward(calc_grads[grad_idx])
                        grad_idx += 1
                
            elif self.is_leaf:
                self.grad = self.grad + grad if self.grad is not None else grad

    def operation(self, op, *inputs):
        requires_grad = False
        inputs = list(inputs)
        for i in range(len(inputs)):
            inputs[i] = inputs[i] if isinstance(inputs[i], Param) else Param(inputs[i])
            requires_grad = requires_grad or inputs[i].requires_grad

        node = Param(op.forward(*inputs), requires_grad=requires_grad)
        node.is_leaf = False

        if requires_grad:
            node.grad_fn = op()
            node.grad_fn.saved_for_backward(*inputs)

        return node
    
    def __getitem__(self, idx):
        node = Param(Get.forward(self, idx), requires_grad=self.requires_grad)
        node.is_leaf = False
        if self.requires_grad:
            node.grad_fn = Get()
            node.grad_fn.saved_for_backward(self, idx)
        return node

    def __setitem__(self, idx, value):
        
        if self.is_leaf and self.requires_grad:
            raise "IDK, But Pytorch denied this condition"
        
        value = value if isinstance(value, Param) else Param(value)

        self.data = Set.forward(self, value, idx)
        self.requires_grad=self.requires_grad or value.requires_grad
        
        if self.requires_grad:
            self.grad_fn = Set(self.grad_fn)
            self.grad_fn.saved_for_backward(self, value, idx)

    def __add__(self, other):
        return self.operation(Add, self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.operation(Sub, self, other)
    
    def __rsub__(self, other):
        return self.operation(Sub, other, self)

    def __mul__(self, other):
        return self.operation(Mul, self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    
    def __truediv__(self, other):
        return self.operation(Div, self, other)
    
    def __rtruediv__(self, other):
        return self.operation(Div, other, self)
    
    def __rfloordiv__(self, other):
        return self.operation(FloorDiv, self, other)
    
    def __rfloordiv__(self, other):
        return self.operation(FloorDiv, other, self)
    
    def __pow__(self, other):
        return self.operation(Pow, self, other)
    
    def __rpow__(self, other):
        return self.operation(Pow, other, self)
    
    def __mod__(self, other):
        return self.operation(Mod, self, other)
    
    def __rmod__(self, other):
        return self.operation(Mod, other, self)
    
    def __abs__(self):
        return self.operation(Abs, self)
    
    def __neg__(self):
        return self.operation(Neg, self)
    
    def __matmul__(self, other):
        return self.operation(MatMul, self, other)
    
    def __rmatmul__(self, other):
        return self.operation(MatMul, other, self)
    
    def __lt__(self, other): 
        if (isinstance(other, Param)):
            return self.data < other.data
        else :
            return self.data < other
        
    def __le__(self, other): 
        if (isinstance(other, Param)):
            return self.data <= other.data
        else :
            return self.data <= other
        
    def __gt__(self, other): 
        if (isinstance(other, Param)):
            return self.data > other.data
        else :
            return self.data > other
        
    def __ge__(self, other): 
        if (isinstance(other, Param)):
            return self.data >= other.data
        else :
            return self.data >= other
    
    def __eq__(self, other):
        if (isinstance(other, Param)):
            return self.data == other.data
        else :
            return self.data == other
        
    def __ne__(self, other):
        if (isinstance(other, Param)):
            return self.data != other.data
        else :
            return self.data != other
            
        