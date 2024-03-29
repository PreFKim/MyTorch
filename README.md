# MyTorch (2024-01-23 ~ 2024-01-30)

**[Velog : [딥러닝] 경사하강법 구현부터 학습까지(1)](https://velog.io/@pre_f_86/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95-%EA%B5%AC%ED%98%84%EB%B6%80%ED%84%B0-%ED%95%99%EC%8A%B5%EA%B9%8C%EC%A7%80-1)**

최근 'SKT FLY AI'에서 공부하며 순전파와 역전파에 대해 강의를 들으며 갑자기 **'순전파 역전파를 전개해본적이 있던가?'** 라는 생각이 들어 계산해보려 하니 간단한 레이어 조차 계산하지 못하였습니다.

과거에도 [PyTorch 독립 레이어 만들기(Grad 끊기)](https://velog.io/@pre_f_86/PyTorch-%EB%8F%85%EB%A6%BD-%EB%A0%88%EC%9D%B4%EC%96%B4-%EB%A7%8C%EB%93%A4%EA%B8%B0Grad-%EB%81%8A%EA%B8%B0) 글을 작성하면서도 코드로 실험을 하며 알았지 정확히 왜 레이어가 단절되는지에 대해서는 정확히 몰랐던 거 같습니다.

경사하강법을 이해하고 순전파와 역전파를 통한 파라미터 학습까지 진행하여 Gradient의 흐름을 이해하고자 진행한 프로젝트입니다.

## How to Use

1. 깃허브를 clone한다.

    git clone https://github.com/PreFKim/MyTorch.git

2. clone한 디렉토리로 이동하여 src를 import해준다.

```python
import src as my
```

3. 원하는 구조에 맞게 모델을 수정한다.(PyTorch와 형식은 비슷합니다.)

```python
def relu(x):
    mul = my.ones(x.shape)
    mul[x<=0] = 0
    x = x*mul
    return x


class mymodel(my.layers.Module):
    def __init__(self):
        self.l1 = my.layers.Linear(2,4)
        self.l2 = my.layers.Linear(4,4)
        self.l3 = my.layers.Linear(4,4)
        self.l4 = my.layers.Linear(4,1)

    def forward(self,x):
        out = self.l1(x)
        out = relu(out)
        out = self.l2(out)
        out = relu(out)
        out = self.l3(out)
        out = relu(out)
        out = self.l4(out)
        return out
```

4. Optimizer와 학습 코드를 작성한다.

```python
x = my.tensor([list(range(100)),list(range(100))]).reshape(-1,2)
y = my.tensor([list(range(100))]).reshape(-1,1)
model = mymodel()
optim = my.optimizers.Adam(params=model.parameters(),lr=1e-3)
# y=x 함수 학습
for i in range(100):
    out = model(x)
    loss = ((out-y)**2).mean()
    loss.backward()
    optim.update()
    optim.zero_grad()
    if (i%10==0):
        print(loss)  
print(loss)
model([1000,1000])
```

### Result

![](./imgs/Train_result.PNG)

## Directory
    MyTorch
        │  .gitignore
        │  README.md
        │  test.ipynb       # 구현 실험 코드
        │  __init__.py
        │
        └─src
            │  parameter.py # Parameter 코드
            │  ...
            │
            ├─layers        # Layer 코드
            │  module.py    # Layer Module 코드
            │  ...
            │
            └─optimizers    # Optimizer 코드
                ...

## 문제점

1. Numpy의 Dtype이 Object 타입으로 Param 클래스를 받기 때문에(동적할당) 실제 연산시 속도가 매우 느려짐
