from src.layers.module import Module
from src.parameter import Param
import numpy as np

class Linear(Module):
    def __init__(self, in_channels, out_channels, bias=True):

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # kaiming uniform init
        # fan = _calculate_correct_fan(tensor, mode) # in_channels
        # gain = calculate_gain(nonlinearity, a) # math.sqrt(2.0 / (1 + a**2)) [a=5] -> 0.2773500981126146
        # std = gain / math.sqrt(fan) # gain / (in_channels) **(1/2)
        # bound = math.sqrt(3.0) * std # 1.7320508075688772 * std
        
        bound = 1.7320508075688772 * 0.2773500981126146/(in_channels)**(1/2)

        self.weight = Param(
            # np.ones((in_channels, out_channels)).astype(np.float32),
            # np.random.uniform(-np.sqrt(6/in_channels), np.sqrt(6/in_channels), size=(in_channels, out_channels)).astype(np.float32), 
            # np.random.normal(loc=0, scale=np.sqrt(2/in_channels), size=(in_channels, out_channels)).astype(np.float32),
            np.random.uniform(-bound, bound, size=(in_channels, out_channels)).astype(np.float32), 
            requires_grad=True
            )

        self.bias = Param(
            # np.ones((out_channels)).astype(np.float32),
            np.random.uniform(-np.sqrt(1/in_channels), np.sqrt(1/in_channels), size=(out_channels)).astype(np.float32),
            requires_grad=True
            ) if bias else None
        
    def forward(self, x):
        if self.bias is not None : return x @ self.weight + self.bias
        else : return x @ self.weight 