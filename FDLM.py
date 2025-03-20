import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import sin,cos,atan2,sqrt,pi
import torch.utils
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import time
import os

TIME_ID = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
IMAGE_DIR = 'images/' + TIME_ID
MODEL_DIR ='models/'
PRINT_EPOCHS = 250
if(not os.path.exists(IMAGE_DIR)):
    os.makedirs(IMAGE_DIR)
if(not os.path.exists(MODEL_DIR)):
    os.makedirs(MODEL_DIR)

# 下面是测试用的理想信号
test_T = np.array([2,5,8,19,40,43,50])  # 周期
test_amplitudes = np.array([1,2,3,4,5,2,7])  # 振幅
test_phases = np.array([1,0,2,1,-2,0,-1])  # 相位

class FDLM(nn.Module):
    def __init__(self, train_data,max_period=None,included_times=None,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        '''
        train_data: 时域上的训练数据，numpy.ndarray或List格式
        max_period: 最大周期，默认为训练数据长度
        included_times: 指定包含的频率分量的周期，在指定范围以外的频率直接被排除
        '''
        super().__init__()
        self.device = device
        # 对train_data消去常数项
        self.train_data = torch.tensor(train_data,dtype=torch.float64,device=device)
        self.max_period = len(train_data) if max_period is None else max_period
        self.periods = torch.arange(2, self.max_period+1, device=device,dtype=torch.float64) # 常数：周期
        self.trainable_amplitudes = nn.Parameter(torch.ones((self.max_period-1,1), device=device,dtype=torch.float64)) # 可训练参数：振幅
        self.trainable_phases = nn.Parameter(torch.randn(self.max_period-1, device=device,dtype=torch.float64)*torch.pi) # 可训练参数：相位
    
        self.frequency_convolution_coefficients_A = None # 频率卷积系数
        self.frequency_convolution_coefficients_B = None # 频率卷积系数
        included_times = np.array(included_times) if included_times is not None else None
        self.frequency_model_coefficients_culculation(included_times) # 计算频率卷积系数
        
        self.mean = torch.mean(self.train_data)  # 暂且认为均值是个常数，不可训练
        self.frequency_train_data = None
        self.frequency_train_data_culculation() # 计算实验采样信号的频率域信号
        self.train_dataset = TensorDataset(self.periods.long().unsqueeze(1),self.frequency_train_data)

    def frequency_train_data_culculation(self):
        ground_truth = self.train_data-self.mean
        # print(ground_truth)
        N= ground_truth.shape[0]
        # 假设N=1000，那么矩阵大小就是N*N=10^6，2GB显存可以容纳2^28=268,435,456个torch.float64，是够用的
        # 若N超过10000则将它拆分为两半分别计算再相加(未实现)
        Tc = self.periods.reshape(-1,1)
        n = torch.arange(N,dtype=torch.float64,device=self.device).unsqueeze(0)
        __matrix = (2*pi/Tc)@n

        ground_truth = ground_truth.reshape(-1,1)
        sin_sum = sin(__matrix)@ground_truth
        cos_sum = cos(__matrix)@ground_truth
        sin_sum[0,:] = 0 # 【特别地】当Tc=2时，sin总和为0
        self.frequency_train_data = torch.cat((sin_sum,cos_sum),dim=1)
        # self.frequency_train_data的维度是：
        #   [[s2,c2],
        #    [s3,c3],
        #     ..... ,
        #    [sN,cN]]
 
    def frequency_model_coefficients_culculation(self,included_times):
        N = self.train_data.shape[0]
        T = self.periods
        #一定要把T转为float64类型，否则会出现精度问题。不能在alpha之后才转精度！
        alpha = pi / T
        alpha = alpha.repeat(self.max_period-1,1)
        gamma = N * alpha
        beta = gamma - alpha

        alpha1_add_alpha2 = alpha + alpha.T
        alpha1_sub_alpha2 = alpha - alpha.T
        gamma1_add_gamma2 = gamma + gamma.T
        gamma1_sub_gamma2 = gamma - gamma.T

        # 计算K1和K2
        SdS_add = sin(gamma1_add_gamma2)/sin(alpha1_add_alpha2)
        SdS_sub = sin(gamma1_sub_gamma2)/sin(alpha1_sub_alpha2)
        SdS_sub.fill_diagonal_(N)  # 处理对角线 $T=T_{c}$ 的情况，使得 sin(γ_1 -γ_2 )/sin(α_1 -α_2 ) = N

        K1_sin = ( SdS_sub - SdS_add )*cos(beta.T)
        K2_sin = ( SdS_add + SdS_sub )*sin(beta.T)
        A_sin = sqrt(K1_sin**2 + K2_sin**2)/2
        B_sin = atan2(K1_sin,K2_sin) + beta
        A_sin[0,:] = 0 # 【特别地】当Tc=2时，sin总和为0

        K1_cos = ( SdS_add - SdS_sub )*sin(beta.T)
        K2_cos = ( SdS_add + SdS_sub )*cos(beta.T)
        A_cos = sqrt(K1_cos**2 + K2_cos**2)/2
        B_cos = atan2(K1_cos,K2_cos) + beta
        
        A = torch.stack((A_sin,A_cos),dim=0)
        # A[torch.abs(A)<1e-10] = 0 # 对于分量极小的频率，也直接消为0 
        B = torch.stack((B_sin,B_cos),dim=0)
        
        if(included_times is not None): # 
            excluded_times = np.setdiff1d(np.arange(2,self.max_period+1),included_times)
            A[:,:,excluded_times-2] = 0
        
        self.frequency_convolution_coefficients_A = A
        self.frequency_convolution_coefficients_B = B
        # print(self.frequency_convolution_coefficients_A.shape)
        # self.frequency_convolution_coefficients_A的维度是：(sin和cos, Tc, T)
        # 它与trainable_amplitudes做矩乘后frequency_output的维度是：(sin和cos, Tc, 1)
        # 而frequency_train_data的维度是：(Tc, sin和cos（不分开）)
        # 所以记得要做维度转换：frequency_output.transpose(0,1).squeeze(2)
    
    
    def __call__(self, t:torch.Tensor):
        # 前向函数仅用于在应用时输出预测值，不用于训练
        # 目前暂时只接受单变量输入，比如[[1],[2],[3]]，表示要分别获取n=1,n=2,n=3的预测值
        if t.dim() == 1: # 如果输入[1,2,3]，自动转为[[1],[2],[3]]
            t = t.unsqueeze(1)
            
        t = t.to(self.device)
        with torch.no_grad():
            t = self.time_model_output(t)
        return t
    
    
    def frequency_model_output(self, Tc:torch.Tensor):
        # 频域上的模型输出，用于训练
        Tc = Tc.squeeze(1)
        # print(Tc)
        coefficients_A = self.frequency_convolution_coefficients_A[:,Tc-2]
        coefficients_B = self.frequency_convolution_coefficients_B[:,Tc-2]
        # print(coefficients_A.shape)
        # print(coefficients_B.shape)
        frequency_output = coefficients_A*sin(coefficients_B+self.trainable_phases)
        frequency_output = frequency_output@self.trainable_amplitudes
        return frequency_output.transpose(0,1).squeeze(2)
    
    def time_model_output(self,t:torch.Tensor):
        t = t*2*torch.pi
        t = t.repeat(1, self.max_period-1)
        t = t / self.periods + self.trainable_phases
        t = torch.sin(t) @ self.trainable_amplitudes
        return t + self.mean
    
    def regularization_loss_func(self,basic_loss_func=torch.nn.MSELoss()):
       
        relu = torch.functional.F.relu
        def trainable_amplitudes_regularization(trainable_amplitudes):
            # 把可训练参数 $A_{T}$ 的取值范围约束在 $[0,+∞]$
            loss = relu( -trainable_amplitudes + 0.5 )
            loss = torch.sum(loss**2)
            # print('amplitudes_regularization_loss:',loss)
            return loss
        
        def trainable_phases_regularization(trainable_phases):
            # 把可训练参数 $φ_{T}$ 的取值范围约束在 $[-π,π]$
            loss = relu(trainable_phases-pi-0.1)+relu(-trainable_phases+pi+0.1)
            loss = torch.sum(loss**2)
            return loss
        
        def frequency_aggregation_loss_function(SCALE):
            # 频率汇聚损失函数，使得频率尽可能集中到单个频率上
            # 具体操作方式就是计算每5个频率之间的信息熵，使得信息熵尽量小
            A = self.trainable_amplitudes.view(1,-1)
            A = relu(A)+relu(-A)  # 把负的振幅变成正的，同时让大的越大，小的越小
            A = torch.log(A+1)
            A_sum =  A.unfold(dimension=1, size=SCALE, step=1).sum(dim=2).squeeze(0)
            A = A.squeeze(0)
            log_sum = torch.log(A_sum)
            log_A = torch.log(A)
            entropy = torch.zeros_like(A_sum,device=self.device)
            # print(amplitudes.shape)
            # print(average.shape)
            for i in range(0,SCALE):
                entropy += (log_A[i:A_sum.shape[0]+i] - log_sum)*A[i:A_sum.shape[0]+i]
            entropy = -torch.sum(entropy) # 别忘了取负号
            return entropy
        
        def amplitudes_to_less(): # 让接近0的值直接跑向0
            loss = F.sigmoid(self.trainable_amplitudes**2) - 0.5
            loss = torch.sum(loss)
            return loss
            
        
        def total_loss_func(y_pred,y_true):
            loss = basic_loss_func(y_pred,y_true)
            loss += trainable_amplitudes_regularization(self.trainable_amplitudes)
            loss += trainable_phases_regularization(self.trainable_phases)
            # loss += 0.1*frequency_aggregation_loss_function(3) # 频率汇聚
            loss += 0.5*frequency_aggregation_loss_function(7) # 频率汇聚
            # loss += amplitudes_to_less() 
            return loss
        
        return total_loss_func
    
    def train(self,epochs=100,batch_size=256):
        dataloader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(),lr=0.05)
        loss_func = self.regularization_loss_func(torch.nn.MSELoss())
        for epoch in range(epochs+1):
            super().train()
            # 在频域上训练
            for Tc,frequency_train_data in dataloader:
                optimizer.zero_grad()
                frequency_pred = self.frequency_model_output(Tc)
                loss = loss_func(frequency_pred,frequency_train_data)
                loss.backward()
                optimizer.step()
            # # 在时域上训练
            # time_pred = self.time_model_output(train_t).squeeze(1)
            # loss = loss_func(time_pred,self.train_data)
            # loss.backward()
            # optimizer.step()
            if epoch%PRINT_EPOCHS == 0:
                torch.save(self.state_dict(),os.path.join(MODEL_DIR ,TIME_ID+'.pth') )
                print('epoch:',epoch,'loss:',loss.item())
                self.draw(epoch)
    
    def draw(self,epoch):
        with torch.no_grad():
            plt.figure(figsize=(25,15))
            data_len = self.train_data.shape[0]
            # 时域
            train_signal = self.train_data.cpu().numpy()
            true_signal = func2sequence(signal_func,-data_len,2*data_len) # 理想真实信号，调试用，记得删
            t = torch.arange(-data_len,2*data_len,dtype=torch.float64,device=self.device).unsqueeze(1) # 测试时用的时间序列
            pred_signal = self.time_model_output(t).squeeze(1).cpu().numpy()
            
            t = np.arange(-data_len,2*data_len)
            plt.subplot(3,1,1)
            # plt.suptitle('epoch:'+str(epoch),fontsize=30)
            # y=0横坐标轴
            plt.plot(t,true_signal,label='true signal',color='green') # 理想真实信号，调试用，记得删
            plt.plot(np.arange(data_len),train_signal,label='train data',color='blue')
            plt.plot(t,pred_signal,label='predict',alpha=0.5,color='red')
            plt.grid(True)
            plt.legend()
            # 频域
            Tc = torch.arange(2,self.max_period+1).unsqueeze(1)
            frequency_pred = self.frequency_model_output(Tc)
            frequency_data = self.frequency_train_data
            sin_pred = frequency_pred[:,0]
            sin_data = frequency_data[:,0]
            # plt.subplot(3,2,3)
            plt.subplot(3,1,2)
            plt.plot(Tc,sin_data.cpu().numpy(),label='sin_data',color="green")
            plt.plot(Tc,sin_pred.cpu().numpy(),label='sin_pred',color="red")
            plt.grid(True)
            plt.legend()
            # cos_pred = frequency_pred[:,1] # cos的频域
            # cos_data = frequency_data[:,1]
            # plt.subplot(3,2,4)
            # plt.plot(Tc,cos_data.cpu().numpy(),label='cos_data',color="green")
            # plt.plot(Tc,cos_pred.cpu().numpy(),label='cos_pred',color="red")
            # plt.legend()
            # 未泄露的模型频域
            plt.subplot(3,1,3)
            plt.plot(Tc,self.trainable_amplitudes.cpu().numpy(),label='model_amplitudes',color="red",marker='o')
            plt.scatter(test_T,test_amplitudes,color="green",marker='*',s=100)
            plt.grid(True)
            plt.legend()
            
            plt.savefig(os.path.join(IMAGE_DIR,str(epoch)+'.png'),bbox_inches='tight')
            plt.close()

def signal_func(t):
    T = test_T
    amplitudes = test_amplitudes
    phases = test_phases
    return np.sum(amplitudes*np.sin(2*np.pi*t/T+phases),axis=0)

def func2sequence(func, begin,end):
    sequence_data = []
    for t in range(begin,end):
        sequence_data.append(func(t))
    return np.array(sequence_data)

train_data = func2sequence(signal_func,0,500)
model = FDLM(train_data,np.max(test_T)+10)
# model = FDLM(train_data,60,included_times=[2,5,8,19,40,43,50])
model.train(epochs=10000,batch_size=32)


