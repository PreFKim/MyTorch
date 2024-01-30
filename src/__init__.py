from .parameter import Param # parameter.py 의 Param 클래스만 가져옴

from .array import (
    Tensor,
    tensor, 
    from_numpy,
    zeros,
    ones
)

from . import layers # layer폴더의 모든 모듈을 가져옴 
from . import optimizers # optimizers폴더의 모든 모듈을 가져옴

