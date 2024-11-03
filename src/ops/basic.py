from src.parameter import operation
from src.gradients.basic import Log

def log(x):
    return operation(Log, x)

def exp(x):
    return 2.718281828459045 ** x