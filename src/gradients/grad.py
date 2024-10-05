
class Grad:
    def __init__(self):
        self.saved_tensors = []

    def __len__(self):
        return len(self.saved_tensors)
    
    def saved_for_backward(self, *nodes):
        self.saved_tensors = nodes

    def backward(self, grad=1):
        raise "Backward operation for this operation isn't implemened"