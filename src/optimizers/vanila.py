class GradientDescent: 
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr
    
    def update(self):
        for p in self.params:
            if p.requires_grad :
                p.data = p.data - self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = 0