import numpy as np
class GradientDescent: 
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            if p.grad is not None :
                p.data = p.data - self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)