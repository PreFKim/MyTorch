from ..parameter import Param
from ..array import Tensor

class Module:
    def __call__(self,x):

        out = self.forward(x)
        return out
    
    def parameters(self):
        ret = []
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, Module) or isinstance(attribute, Tensor):
                ret.extend(attribute.parameters())
            elif isinstance(attribute, Param) :
                ret.append(attribute)

        return ret