from src.optimizers import GradientDescent
import numpy as np

class Adam(GradientDescent):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, maximize=False):
        self.params = params
        self.lr = lr
        self.betas = betas 
        self.eps = eps 
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        if amsgrad:
            self.v_hat_max = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None :
                if np.any(np.isnan(p.grad)) or np.any(np.isinf(p.grad)):
                    p.grad = np.nan_to_num(p.grad, nan=0.0, posinf=1e16, neginf=-1e16)

                if self.maximize:
                    gt = -p.grad
                else:
                    gt = p.grad

                
                if self.weight_decay != 0:
                    gt = gt + self.weight_decay * p.data
                
                self.m[i] = self.betas[0]*self.m[i]+(1-self.betas[0])*gt
                self.v[i] = self.betas[1]*self.v[i]+(1-self.betas[1])*(gt)**2

                m_hat = self.m[i]/(1-self.betas[0]) 
                v_hat = self.v[i]/(1-self.betas[1])

                if self.amsgrad:
                    self.v_hat_max[i] = np.maximum(self.v_hat_max[i], v_hat)
                    p.data -= self.lr*m_hat / (self.v_hat_max[i]**(1/2)+self.eps)
                else:
                    p.data -= self.lr*m_hat / (v_hat**(1/2)+self.eps)