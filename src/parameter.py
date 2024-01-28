import numpy as np

class Param:
    def __init__(self,data=0.0,type=0.0,requires_grad=True):
        self.data = data # 가중치 혹은 결과 값
        self.requires_grad = requires_grad
        self.backward_grad = 0.0 # Backpropagtion된 Grad
        self.privious_node = [] # 이전 노드들
        self.foward_grad = [] # 이전 노드들에 대한 foward 가중치
        self.type = type # 0이면 가중치 1이면 노드


    def __repr__(self):
        return f"({self.data}, type:{'Node' if self.type else 'Param'}, foward_grad:{self.foward_grad}, backward_grad:{self.backward_grad} requrired_grad:{self.requires_grad})"
    
    def __add__(self,other):
        node = Param(type=1)
        node.privious_node.append(self)
        node.foward_grad.append(1) # x+a (derivative) -> 1 | x = self

        if (type(node)==type(other)):
            node.data = self.data + other.data
            node.privious_node.append(other)
            node.foward_grad.append(1) # a+x (derivative) -> 1 | x = other 
        else :
            node.data = self.data + other
        
        return node
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self,other):
        node = Param(type=1)
        node.privious_node.append(self)
        node.foward_grad.append(1) # x-a (derivative) -> 1 | x = self

        if (type(node)==type(other)):
            node.data = self.data - other.data
            node.privious_node.append(other)
            node.foward_grad.append(-1) # a - x (derivative) -> -1 | x = other
        else :
            node.data = self.data - other
        
        return node
    
    def __rsub__(self,other):
        node = Param(type=1)
        node.privious_node.append(self)
        node.foward_grad.append(-1) # a - x (derivative) -> -1 | x = self

        if (type(node)==type(other)):
            node.data = other.data - self.data 
            node.privious_node.append(other)
            node.foward_grad.append(1) # x - a (derivative) -> 1 | x = other
        else :
            node.data = other - self.data
        return node

    
    def __mul__(self,other):
        node = Param(type=1)
        node.privious_node.append(self)

        if (type(node)==type(other)):
            node.data = self.data * other.data

            node.foward_grad.append(other.data) # x*a (derivative) -> a | x = self

            node.privious_node.append(other)
            node.foward_grad.append(self.data) # a*x (derivative) -> a | x = other
        else :
            node.foward_grad.append(other) # x*a (derivative) -> a | x = self
            node.data = self.data * other
        return node
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    
    def __truediv__(self,other):
        node = Param(type=1)
        node.privious_node.append(self)

        if (type(node)==type(other)):
            node.data = self.data / other.data

            node.foward_grad.append(1/other.data) # x/a = 1/a * x (derivative) -> 1/a | x = self

            node.privious_node.append(other)
            node.foward_grad.append( self.data / (other.data ** 2)) # a/x  = ax^(-1) (derivative)-> -ax^(-2) | x = other
        else :
            node.foward_grad.append(1/other) # x/a = 1/a * x (derivative) -> 1/a | x = self
            node.data = self.data / other
        return node
    
    def __rtruediv__(self,other):
        node = Param(type=1)
        node.privious_node.append(self)

        if (type(node)==type(other)):
            node.data = other.data / self.data

            node.foward_grad.append(other.data / (self.data ** 2)) # a/x  = ax^(-1) (derivative)-> -ax^(-2) | x = self

            node.privious_node.append(other)
            node.foward_grad.append( 1 / self.data ) # x/a = 1/a * x (derivative) -> 1/a | x = other
        else :
            node.foward_grad.append(other / (self.data ** 2)) # a/x  = ax^(-1) (derivative)-> -ax^(-2) | x = self
            node.data = other / self.data
        return node
    
    def __pow__(self,other):
        node = Param(type=1)
        node.privious_node.append(self)

        if (type(node)==type(other)):
            node.data = self.data ** other.data

            node.foward_grad.append(other.data * self.data ** (other.data-1)) # x**a (derivative) -> a* (x**(a-1)) | x = self


            node.privious_node.append(other)
            node.foward_grad.append(np.log(self.data)*self.data ** other.data) # a**x (derivative) -> ln(a)* (a**x) | x = other
        else :
            node.foward_grad.append(other * self.data ** (other-1)) # x**a (derivative) -> a*(x**(a-1)) | x = self
            node.data = self.data ** other

        return node
    
    def __rpow__(self,other):
        node = Param(type=1)
        node.privious_node.append(self)

        if (type(node)==type(other)):

            node.data = other.data ** self.data
            
            node.foward_grad.append(np.log(other.data)*other.data ** self.data) # a**x (derivative) -> ln(a)* (a**x) | x = self

            node.privious_node.append(other)
            node.foward_grad.append(self.data * other.data ** (self.data-1)) # x**a (derivative) -> a*(x**(a-1)) | x = other
        else :
            node.foward_grad.append(np.log(other)*other ** self.data) # a**x (derivative) -> ln(a)* (a**x) | x = self
            node.data = other ** self.data

        return node
    
    def __floordiv__(self,other):
        node = Param(type=1,requires_grad=False)
        
        if (type(self)==type(other)):
            node.data = self.data // other.data
        else :
            node.data = self.data // other
        return node
    
    def __mod__(self,other):
        node = Param(type=1,requires_grad=False)

        if (type(self)==type(other)):
            node.data = self.data % other.data
        else :
            node.data = self.data % other
        return node
    
    def __abs__(self):
        node = Param(type=1)
        node.privious_node.append(self)
        if (self.data>=0):
            node.data = self.data
            node.foward_grad.append(1)
        else:
            node.data = -self.data
            node.foward_grad.append(-1)
        return node

    def print_node(self,depth=0):
        s = ""
        for i in range(depth):
            s+= '\t'
        print(s + self.__repr__() + '(')
        for p in self.privious_node:
            p.print_node(depth+1)
        print(s+')')

        