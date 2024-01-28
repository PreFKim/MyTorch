class Origin: 
    def __init__(self,params,lr=1e-3):
        self.params = params
        self.lr = lr
    
    def update(self):
        for i,p in enumerate(self.params):
            if p.requires_grad :
                p.data = p.data - self.lr*p.backward_grad

    def zero_grad(self):
        for p in self.params:
            p.backward_grad = 0