from ..parameter import Param

class Module:
    def __call__(self,x):

        out = self.forward(x)
        return out
    
    def parameters(self):
        ret = []
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, Module):
                params = attribute.parameters()
                ret += params
            elif isinstance(attribute, Param) :
                ret.append(params)


        return ret