class Module:

    def forward(self,x):
        return x
    
    def __call__(self,x):

        out = self.forward(x)
        return out