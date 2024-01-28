class Adam:
    def __init__(self,params,lr=1e-3,betas=[0.9,0.999],eps=1e-10):
        self.params = params
        self.lr = lr
        self.betas = betas # [Momentum의 지수이동 평균,RMSProp의 지수이동평균]
        self.eps = eps # 분모가 0이되지 않도록 설정

        self.m = [0 for _ in range(len(params))]
        self.g = [0 for _ in range(len(params))]
    
    def update(self):
        for i,p in enumerate(self.params):
            if p.requires_grad :
                self.m[i] = self.betas[0]*self.m[i]+(1-self.betas[0])*p.backward_grad
                self.g[i] = self.betas[1]*self.g[i]+(1-self.betas[1])*(p.backward_grad)**2


                # 첫 학습의 경우 m은 0이고 초기 Grad가 0에 가깝다면 m 값이 작아 학습이 안될 가능성이 있음
                m_hat = self.m[i]/(1-self.betas[0]) # 1-betas의 값은 작아지기 때문에 m_hat이 커질 수 있음 
                g_hat = self.g[i]/(1-self.betas[1])
                

                p.data = p.data - self.lr*m_hat/(g_hat+self.eps)**(1/2)

    def zero_grad(self):
        for p in self.params:
            p.backward_grad = 0