class Module:

    def forward(self,x):
        return x
    
    def __call__(self,x):

        out = self.forward(x)
        return out
    
    def parameters(self):
        ret = []
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, Module):                 
                params = attribute.parameters()
                ret += params

        return ret