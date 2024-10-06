# MyTorch (2024-01-23 ~ 2024-01-30)

**[Velog : [딥러닝] 경사하강법 구현부터 학습까지](https://velog.io/@pre_f_86/series/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95-%EA%B5%AC%ED%98%84%EB%B6%80%ED%84%B0-%ED%95%99%EC%8A%B5%EA%B9%8C%EC%A7%80)**

**[Velog : [PyTorch] AutoGrad란 무엇인가?](https://velog.io/@pre_f_86/series/PyTorch-AutoGrad%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80)**

최근 'SKT FLY AI'에서 공부하며 순전파와 역전파에 대해 강의를 들으며 갑자기 **'순전파 역전파를 전개해본적이 있던가?'** 라는 생각이 들어 계산해보려 하니 간단한 레이어 조차 계산하지 못하였습니다.

과거에도 [PyTorch 독립 레이어 만들기(Grad 끊기)](https://velog.io/@pre_f_86/PyTorch-%EB%8F%85%EB%A6%BD-%EB%A0%88%EC%9D%B4%EC%96%B4-%EB%A7%8C%EB%93%A4%EA%B8%B0Grad-%EB%81%8A%EA%B8%B0) 글을 작성하면서도 코드로 실험을 하며 알았지 정확히 왜 레이어가 단절되는지에 대해서는 정확히 몰랐던 거 같습니다.

경사하강법에 대해 이해하고 그래디언트의 흐름을 파악하는 것이 주요 목표이며, 외부 프레임워크를 최대한 쓰지 않고 직접 순전파 역전파를 구현하고 학습까지 진행하고자 한 프로젝트입니다.

# Todo

- [ ] Optimize the code of the Operation method in parameter.py
- [x] Implements the array manipulation operation(ones, zeros etc...)
- [x] Implements the convolution layer
- [x] Implements the Non-Linear functions
- [ ] Train/inference with real dataset(Iris, MNIST)
- [ ] Implements custom array object

# How to Use

1. 깃허브를 clone한다.

    ```bash
    git clone https://github.com/PreFKim/MyTorch.git
    pip install numpy
    ```

2. clone한 디렉토리로 이동하여 src를 import해준다.

    ```python
    import src as my
    import numpy as np
    ```

3. 원하는 구조에 맞게 모델을 수정한다.(PyTorch와 형식은 비슷합니다.)

    ```python
    def relu(x):
        x[x<0] = 0
        return x

    class mymodel(my.layers.Module):
        def __init__(self):
            self.l1 = my.layers.Linear(2,16,bias=True)
            self.l2 = my.layers.Linear(16,16,bias=True)
            self.l3 = my.layers.Linear(16,16,bias=True)
            self.l4 = my.layers.Linear(16,1,bias=True)

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
    x = my.Param(np.stack([np.arange(100), np.arange(100)], 1))/100
    y = my.Param(np.arange(100).reshape(-1, 1))/100

    model = mymodel()
    optim = my.optimizers.Adam(params=model.parameters(),lr=1e-3)
    for i in range(1000):
        out = model(x)
        
        loss = (out-y)**2

        loss_mean = 0
        for l in loss:
            loss_mean = loss_mean + l/100
        loss_mean.backward()

        optim.update()
        optim.zero_grad()
        if (i%10==0):
            print(loss_mean)  
    model([1000,1000])
    ```     

## Result

![](./imgs/Train_result.PNG)

# Directory
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
            ├─gradients     # Gradient 코드
            │  basic.py     # 기본 연산 코드
            │  ...
            │
            ├─layers        # Layer 코드
            │  module.py    # Layer Module 코드
            │  ...
            │
            └─optimizers    # Optimizer 코드
                ...
