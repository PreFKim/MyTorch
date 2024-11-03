from src.parameter import Param

class Module:
    def __call__(self,x):
        out = self.forward(x)
        return out
    
    def parameters(self):
        ret = []
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, Module):
                ret.extend(attribute.parameters())
            elif isinstance(attribute, Param) :
                ret.append(attribute)

        return ret