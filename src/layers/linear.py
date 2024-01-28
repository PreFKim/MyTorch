from ..parameter import Param
from .module import Module

class Linear(Module):
    def __init__(self,in_channels,out_channels,bias=True):

        self.w = [[Param(1) for j in range(in_channels) ] for i in range(out_channels)] # out_channels, in_channels
        

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bias = bias

        if self.bias:
            self.b = [Param(1) for i in range(out_channels)] # out_channels

    def forward(self,x):

        ret = []
        for i in range(self.out_channels):
            sum = x[0] * self.w[i][0]
            for j in range(1,self.in_channels):
                sum = sum + self.w[i][j] * x[j]
            ret.append(sum)

        if self.bias:
            for i in range(self.out_channels):
                ret[i] = ret[i] + self.b[i]
        
        return ret
    
    def parameters(self):
        ret = []
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                ret.append(self.w[i][j])

        if self.bias:
            for i in range(self.out_channels):
                ret.append(self.b[i])
        return ret
    


