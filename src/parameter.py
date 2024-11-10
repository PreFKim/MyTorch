import numpy as np
from src.gradients.grad import GradFunction, ContextManager, Accumulate
from src.gradients.basic import *
from src.gradients.index import Get, Set
from src.gradients.manipulate import Reshape

def operation(op, *inputs, convert=True, inplace=False): # role for torch.autograd.Function().apply()
    requires_grad = False
    next_functions = []
    ctx = ContextManager()

    if convert: # tuple -> list for writable(wraping)
        inputs = list(inputs)

    for i in range(len(inputs)):
        if isinstance(inputs[i], Param)==False and convert:
            inputs[i] = Param(inputs[i])

        if isinstance(inputs[i], Param):
            grad_fn = None
            if inputs[i].requires_grad:
                requires_grad = True
                if inputs[i].is_leaf:
                    grad_fn = Accumulate(inputs[i])
                else:
                    grad_fn = inputs[i].grad_fn # None or GradFn
            next_functions.append(grad_fn)

    if inplace:
        node = inputs[0]
        node.data = op.forward(ctx, *inputs)
        node.requires_grad = requires_grad
        node._version += 1 
    else:
        output = op.forward(ctx, *inputs)
        node = Param(output, requires_grad=requires_grad)
        node.is_leaf = False

    if requires_grad:
        node.grad_fn = GradFunction(op, ctx, next_functions)

    return node

class Param:
    def __init__(self, data, dtype=None, requires_grad=False):
        if isinstance(data, (np.ndarray, np.generic)): # Numpy Array or Numpy Scalar
            self.data = data if dtype is None else data.astype(dtype)
        elif isinstance(data, (int, float, complex, list, tuple)) :
            self.data = np.array(data, dtype=dtype if dtype is not None else np.float32)
        else :
            raise ValueError("Not supported data type")
        
        self.grad = None
        self.grad_fn = None
        self.requires_grad = requires_grad
        self.is_leaf = True
        self._version = 0

    def __repr__(self):
        return f"Data:{self.data}"+ (f", requrired_grad:{self.requires_grad}" if self.requires_grad else "")
    
    def __len__(self):
        return self.shape[0]
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def backward(self, grad=1):
        if self.grad_fn is not None:
            if not isinstance(grad, (np.ndarray, np.generic)) :
                grad = np.ones_like(self.data) * grad
            self.grad_fn.backward(grad)

    def __getitem__(self, idx):
        return operation(Get, self, idx, convert=False)
        

    def __setitem__(self, idx, value):
        if self.is_leaf and self.requires_grad:
            raise RuntimeError("IDK, But Pytorch denied this condition")
        value = value if isinstance(value, Param) else Param(value)
        operation(Set, self, value, idx, convert=False, inplace=True)
    
    def clone(self):
        output = Param(self.data, dtype=self.dtype, requires_grad=self.requires_grad)
        output.grad = self.grad 
        output.grad_fn = self.grad_fn
        output.is_leaf = self.is_leaf
        output._version = self._version
        return output

    def __add__(self, other):
        return operation(Add, self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return operation(Sub, self, other)
    
    def __rsub__(self, other):
        return operation(Sub, other, self)

    def __mul__(self, other):
        return operation(Mul, self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return operation(Div, self, other)
    
    def __rtruediv__(self, other):
        return operation(Div, other, self)
    
    def __rfloordiv__(self, other):
        return operation(FloorDiv, self, other)
    
    def __rfloordiv__(self, other):
        return operation(FloorDiv, other, self)
    
    def __pow__(self, other):
        return operation(Pow, self, other)
    
    def __rpow__(self, other):
        return operation(Pow, other, self)
    
    def __mod__(self, other):
        return operation(Mod, self, other)
    
    def __rmod__(self, other):
        return operation(Mod, other, self)
    
    def __abs__(self):
        return operation(Abs, self)
    
    def __neg__(self):
        return operation(Neg, self)
    
    def __matmul__(self, other):
        return operation(MatMul, self, other)
    
    def __rmatmul__(self, other):
        return operation(MatMul, other, self)
    
    def sum(self, dim=None, keepdim=False):
        assert keepdim == False or (keepdim and dim is not None), "If you set keepdim to True, you should input the dim variable"
        return operation(Sum, self, dim, keepdim, convert=False)

    def mean(self, dim=None, keepdim=False):
        assert keepdim == False or (keepdim and dim is not None), "If you set keepdim to True, you should input the dim variable"
        return operation(Mean, self, dim, keepdim, convert=False)
    
    def reshape(self, shape):
        return operation(Reshape, self, shape, convert=False)
        
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
            
        