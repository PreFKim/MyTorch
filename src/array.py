import numpy as np
from .parameter import Param

class Tensor(np.ndarray):
    def __new__(cls, shape, dtype='O'):
        # np.ndarray는 __init__ 메서드에서 초기화 하지 않음
        obj = super(Tensor, cls).__new__(cls, shape, dtype=dtype)
        return obj
    
    def parameters(self):
        ret = []
        myparam = np.reshape(self,-1)
        for p in myparam:
            if isinstance(p,Param):
                ret.append(p)
        return ret
    
    def backward(self,grad=1):
        myparam = np.reshape(self,-1)
        for p in myparam:
            if isinstance(p,Param):
                for i in range(len(p.privious_node)):
                    p.privious_node[i].backward_grad += p.foward_grad[i]*grad
                    p.privious_node[i].backward(p.foward_grad[i]*grad)

    
def get_shape(lst): 
    ret_shape = []
    
    if isinstance(lst,list):
        child = []
        ret_shape.append(len(lst))

        for i,data in enumerate(lst):
            child.append(get_shape(data))
            if (i>0):   
                assert (child[i-1]==child[i]), "Tensor should have same length"
                
        ret_shape.extend(child[0])
    
    return ret_shape 
    

def tensor(data):
    shape = tuple(get_shape(data))
    def replace_value(arr,replace_arr):
        for i in range(len(arr)):
            if isinstance(arr[i],Tensor):
                arr[i] = replace_value(arr[i],replace_arr[i])
            else:
                arr[i] = Param(replace_arr[i])
        return arr
    return replace_value(Tensor(shape),data)

def from_numpy(arr):
    ret = Tensor(arr.shape)
    ret[:] = arr
    def convert_to_param(arr):
        for i in range(len(arr)):
            if isinstance(arr[i],np.ndarray):
                arr[i] = convert_to_param(arr[i])
            else:
                arr[i] = Param(arr[i])
        return arr
    return convert_to_param(ret)

def zeros(shape):
    ret = from_numpy(np.zeros(shape))
    return ret

def ones(shape):
    ret = from_numpy(np.ones(shape))
    return ret


# from collections import deque


# def init_tensor(lst): # 고차원 list -> 1차원 list, 고차원 list의 shape
#     ret_lst = []
#     ret_shape = []
    
#     if isinstance(lst,list):
#         child = []
#         ret_shape.append(len(lst))

#         for i,data in enumerate(lst):
#             child.append(init_tensor(data))
#             ret_lst.extend(child[i][0])
#             if (i>0):   
#                 assert (child[i-1][1]==child[i][1]), "Tensor should have same length"
                
#         ret_shape.extend(child[0][1])
#     else :
#         ret_lst.append(lst)
    
#     return [ret_lst, ret_shape] # list,list 형태

# class Tensor:
#     def __init__(self,shape):
#         self.shape = shape

#         self.size = 1
#         for s in self.shape:
#             self.size = self.size * s

#         self.data = [0 for _ in range(self.size)] # list는 정적 할당이 되지 않는 문제점이 있어 C언어를 연계 해야함
    
#     def __repr__(self):
#         return f"{self.data}, {self.shape}"
    
#     def __getitem__(self,idx): 
#         # idx가 [0,1,:]형식으로 들어오면 (0, 1, slice(None, None, None))으로 들어옴
#         # slice(None, None, None) = start, stop, step

#         assert len(idx)<=len(self.shape), "Too many index"


#         shape = []

#         target_idx = []

#         for i,index in enumerate(len(self.shape)):
        
#             if i < len(idx):
#                 if isinstance(index,slice):
#                     start = 0 if index.start is None else index.start
#                     stop = self.shape[i] if index.stop is None else index.stop
#                     step = 1 if index.step is None else index.step

#                     if start<0: start = max(start + self.shape[i],0)
#                     if stop<0: stop = max(stop + self.shape[i],0)

#                     start = min(self.shape[i],start)
#                     stop = min(self.shape[i],stop)

#                     target_idx.append(list(range(start,stop,step)))
#                     shape.append(len(target_idx[i]))
#                 elif isinstance(index,list):
#                     target_idx.append(index)
#                     shape.append(len(target_idx[i]))
#                 else:
#                     target_idx.append([index]) 
#                     #shape.append(len(target_idx[i])) # 단일 인덱싱은 Shape 처리 안함
#             else:
#                 target_idx.append(list(range(len(self.shape[i]))))
#                 shape.append(len(target_idx[i]))


#         ratio = [1 for _ in range(len(self.shape))] #각 축에 대한 인덱스 크기
#         for i in range(len(self.shape)-2,-1,-1):
#             ratio[i] = (ratio[i+1] * self.shape[i+1]) 
        
#         ret_lst = [] # bfs 알고리즘을 이용한 고차원 인덱스-> 1차원 인덱스 전환
#         q = deque()
#         for i in target_idx[0]:
#             q.append([0,i*ratio[0]])

#         while q:
#             data = q.popleft()
#             if data[0] == len(self.shape)-1:
#                 ret_lst.append(self.data[data[1]])
#             else:
#                 for i in target_idx[data[0]+1]:
#                     q.append([data[0]+1,data[1]+i*ratio[data[0]+1]])

#         ret = Tensor(tuple(shape))
#         ret.data = ret_lst
#         return ret