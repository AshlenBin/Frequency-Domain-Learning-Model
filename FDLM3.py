import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import sin, cos, atan2, sqrt, pi
import torch.utils
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import time
import os

"""
根据傅里叶变换的结果来预设频率分量
只能输入一条序列，不支持同时训练多条序列
"""

TIME_ID = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
IMAGE_DIR = "images/"  # + TIME_ID
MODEL_DIR = "models/"
PRINT_EPOCHS = 500
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


class FDLM(nn.Module):
    def __init__(
        self,
        train_data,
        frequencies_num=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        load_model_path=None
    ):
        """
        train_data: 时域上的训练数据，numpy.ndarray或List格式
        """
        super().__init__()
        self.device = device
        # 对train_data消去常数项
        train_data = torch.as_tensor(train_data, dtype=torch.float64, device=device)
        self.mean = torch.mean(train_data)
        self.train_data = train_data - self.mean # 给模型的训练数据在用户输入的数据基础上去掉均值，在输出时再加上
        self.sequence_len = len(train_data)
        self.frequencies = None  # 预定频率分量
        if(frequencies_num is None):
            frequencies_num = int(len(train_data)/2) 
        self.frequencies_num = None  # 预定频率分量个数
        #【注意】self.frequencies_num并不等于frequencies_num，详情请见init_frequencies()
        self.init_frequencies(frequencies_num)  # 初始化预定频率分量
        # self.frequencies_num = len(test_T)
        # self.frequencies = torch.tensor(1/test_T,dtype=torch.float64,device=self.device)
        self.trainable_amplitudes = nn.Parameter(
            torch.ones(self.frequencies_num, device=device, dtype=torch.float64)
        )  # 可训练参数：振幅
        self.trainable_phases = nn.Parameter(
            torch.ones(self.frequencies_num, device=device, dtype=torch.float64)
        )  # 可训练参数：相位

        self.frequency_convolution_coefficients_A = None  # 频率卷积系数
        self.frequency_convolution_coefficients_B = None  # 频率卷积系数

        self.frequency_model_coefficients_culculation()  # 计算频率卷积系数
        
        if load_model_path is not None:
            self.load_state_dict(torch.load(load_model_path))
            print("load model from: ", load_model_path)


    def init_frequencies(self, frequencies_num):
        density_factor = 1
        fft_pred = torch.fft.rfft(self.train_data)
        fft_amplitudes = torch.abs(fft_pred)*2 / self.sequence_len
        fft_amplitudes = fft_amplitudes.cpu()
        k = torch.arange(1,len(fft_amplitudes),dtype=torch.long)
        threshold = torch.mean(fft_amplitudes)*0.5  # 频率强度低于均值*0.5的将被排除，也就是认为这个频率分量不存在
        
        distributed_ratios = torch.sqrt(fft_amplitudes[:-1]*fft_amplitudes[1:])
        distributed_ratios = distributed_ratios[fft_amplitudes[1:]>threshold]
        k = k[fft_amplitudes[1:]>threshold] 
        distributed_ratios = distributed_ratios / torch.sum(distributed_ratios)
        distributed_nums = distributed_ratios * frequencies_num * density_factor
        # 【注意】self.frequencies_num并不等于frequencies_num，因为取整操作会减少一点数值
        distributed_nums = torch.floor(distributed_nums).type(torch.int)
        distributed_nums[distributed_nums>10] = 10
        distributed_nums[:] = 0
        k_list = []
        for i in range(1,len(distributed_nums)):
            k_list.append(self.frequencies_distribute_func(
                k[i]-1,k[i],
                fft_amplitudes[k[i]-1],fft_amplitudes[k[i]],
                distributed_nums[i])
            )
        
        k = torch.concatenate(k_list).to(self.device)
        self.frequencies = k / self.sequence_len
        self.frequencies_num = len(self.frequencies)
        # 绘制散点分布图
        # fft_amplitudes = fft_amplitudes.numpy()
        # plt.figure(figsize=(15,5))
        # plt.scatter(np.arange(len(fft_amplitudes)),fft_amplitudes,s=10)
        # plt.vlines(np.arange(len(fft_amplitudes)),ymin=0,ymax=fft_amplitudes,colors="blue",linestyles="dotted",alpha=0.4)
        # plt.scatter(k,np.zeros_like(k),s=5)
        # plt.axhline(y=threshold.numpy(), color='gray', linestyle='--')
        # plt.grid(True)
        # plt.show()

    def frequencies_distribute_func(self,k1,k2,A1,A2,distributed_num):
        k1 = torch.as_tensor(k1,dtype=torch.float64)
        k2 = torch.as_tensor(k2,dtype=torch.float64)
        A1 = torch.as_tensor(A1,dtype=torch.float64)
        A2 = torch.as_tensor(A2,dtype=torch.float64) 
        # k = A1/(A1+A2) 
        # 保证分布函数f(k)=0.5，假如k=0.3，那么会有30%的分布点靠近k1，70%的分布点靠近k2
        # 但是为了简化计算，省略掉k
        ratio = A2/A1
        if(distributed_num==1 or ratio<0.01 or ratio>100):
            pass
        a = ratio**2
        t = torch.linspace(0,1,distributed_num+2,dtype=torch.float64)[1:] # 含k2不含k1
        t = torch.log((a-1)*t+1)/torch.log(a)  # 【关键】计算周期分布
        t = k1 + (k2-k1)*t
        return t

    def frequency_train_data_culculation(self, time_train_data):
        # print(ground_truth)
        # 假设N=1000，那么矩阵大小就是N*N=10^6，2GB显存可以容纳2^28=268,435,456个torch.float64，是够用的
        # 若N超过10000则将它拆分为两半分别计算再相加(未实现)
        n = torch.arange(self.sequence_len, dtype=torch.float64, device=self.device)
        __matrix = (2*pi* self.frequencies).unsqueeze(1) @ n.unsqueeze(0)

        time_train_data = time_train_data.unsqueeze(1)
        sin_sum = sin(__matrix) @ time_train_data
        cos_sum = cos(__matrix) @ time_train_data
        sin_sum[0, :] = 0  # 【特别地】当Tc=2时，sin总和为0
        return torch.cat((sin_sum, cos_sum), dim=1)
        # frequency_train_data的维度是：
        #   [[s2,c2],
        #    [s3,c3],
        #     ..... ,
        #    [sN,cN]]

    def frequency_model_coefficients_culculation(self):
        N = self.sequence_len
        # 一定要把T转为float64类型，否则会出现精度问题。不能在alpha之后才转精度！
        alpha = pi * self.frequencies
        alpha = alpha.repeat(self.frequencies_num, 1)
        gamma = N * alpha
        beta = gamma - alpha

        alpha1_add_alpha2 = alpha + alpha.T
        alpha1_sub_alpha2 = alpha - alpha.T
        gamma1_add_gamma2 = gamma + gamma.T
        gamma1_sub_gamma2 = gamma - gamma.T

        # 计算K1和K2
        SdS_add = sin(gamma1_add_gamma2) / sin(alpha1_add_alpha2)
        SdS_sub = sin(gamma1_sub_gamma2) / sin(alpha1_sub_alpha2)
        SdS_sub.fill_diagonal_(N)  # 处理对角线 $T=T_{c}$ 的情况，使得 sin(γ_1 -γ_2 )/sin(α_1 -α_2 ) = N

        K1_sin = (SdS_sub - SdS_add) * cos(beta.T)
        K2_sin = (SdS_add + SdS_sub) * sin(beta.T)
        A_sin = sqrt(K1_sin**2 + K2_sin**2) / 2
        B_sin = atan2(K1_sin, K2_sin) + beta
        A_sin[0, :] = 0  # 【特别地】当Tc=2时，sin总和为0

        K1_cos = (SdS_add - SdS_sub) * sin(beta.T)
        K2_cos = (SdS_add + SdS_sub) * cos(beta.T)
        A_cos = sqrt(K1_cos**2 + K2_cos**2) / 2
        B_cos = atan2(K1_cos, K2_cos) + beta

        A = torch.stack((A_sin, A_cos), dim=0)
        # A[torch.abs(A)<1e-10] = 0 # 对于分量极小的频率，也直接消为0
        B = torch.stack((B_sin, B_cos), dim=0)

        self.frequency_convolution_coefficients_A = A
        self.frequency_convolution_coefficients_B = B
        # print(self.frequency_convolution_coefficients_A.shape)
        # self.frequency_convolution_coefficients_A的维度是：(sin和cos, Tc, T)
        # 它与trainable_amplitudes做矩乘后frequency_output的维度是：(sin和cos, Tc, 1)
        # 而frequency_train_data的维度是：(Tc, sin和cos（不分开）)
        # 所以记得要做维度转换：frequency_output.transpose(0,1).squeeze(2)
        sin_coefficients_A = self.frequency_convolution_coefficients_A[1].cpu().numpy()
        frequencies = self.frequencies.cpu().numpy()
        id = 150
        plt.figure(figsize=(20,5))
        plt.scatter(frequencies,sin_coefficients_A[id],s=10)
        plt.axvline(frequencies[id], color='red', linestyle='--')
        plt.grid(True)
        plt.show()
        exit()

    def __call__(self, t: torch.Tensor):
        # 前向函数仅用于在应用时输出预测值，不用于训练
        # 目前暂时只接受单变量输入，比如[[1],[2],[3]]，表示要分别获取n=1,n=2,n=3的预测值
        if t.dim() == 1:  # 如果输入[1,2,3]，自动转为[[1],[2],[3]]
            t = t.unsqueeze(1)

        t = t.to(self.device)
        with torch.no_grad():
            t = self.model_output_TIME(t)
        return t

    def model_output_FREQUENCY(self, frequencies_id: torch.Tensor):
        # 频域上的模型输出，用于训练
        # 【注意】frequencies_id是频率值在self.frequencies中的下标，不是频率值本身
        if(frequencies_id.dim() == 1):
            frequencies_id = frequencies_id.unsqueeze(1)
        coefficients_A = self.frequency_convolution_coefficients_A[:, frequencies_id]
        coefficients_B = self.frequency_convolution_coefficients_B[:, frequencies_id]
        # print(coefficients_A.shape)
        # print(coefficients_B.shape)
        # print(self.trainable_phases.shape)
        amplitudes = self.trainable_amplitudes
        phases = self.trainable_phases

        frequency_output = coefficients_A * sin(coefficients_B + phases)
        frequency_output = frequency_output @ amplitudes.reshape(-1, 1)
        # print(frequency_output.transpose(0, 1).squeeze(2).squeeze(2).shape)
        return frequency_output.transpose(0, 1).squeeze(2).squeeze(2)

    def model_output_TIME(self, t: torch.Tensor):
        # t的维度长这样：[[1],[2],[3]]
        t = t * 2 * torch.pi
        t = t.repeat(1, self.frequencies_num)
        t = t * 2*pi*self.frequencies + self.trainable_phases
        t = torch.sin(t) @ self.trainable_amplitudes.reshape(-1, 1)
        return t + self.mean

    def regularization_loss_func(self, basic_loss_func=torch.nn.MSELoss()):

        relu = torch.functional.F.relu

        def frequency_aggregation_loss_function(SCALE):
            # 频率汇聚损失函数，使得频率强度尽可能集中到单个频率上
            kernel = torch.arange(1, SCALE + 1, dtype=torch.float64, device=self.device)  # SCALE=5 则 kernel=[1,2,3,4,5]
            A = self.trainable_amplitudes.unsqueeze(0)
            A = torch.abs(A)  # 把负的振幅变成正的
            mean = A.detach().mean(dim=1,keepdim=True) 
            A = mean*torch.tanh(A/(0.8*mean)) # 奖励压低零散的小值，但不奖励拉升集中的大值
            A_sum = A.unfold(dimension=1, size=SCALE, step=1).sum(dim=2)
            p = A.unfold(dimension=1, size=SCALE, step=1) / A_sum.unsqueeze(2)
            pA_sum = p @ kernel.unsqueeze(1)
            distance = kernel.reshape(1,1,-1) - pA_sum
            distance = 0.5 * (distance ** 2) + 1
            loss = -torch.sum(p / distance)
            return loss

        def amplitudes_to_less():  # 让接近0的值直接跑向0
            loss = F.sigmoid(self.trainable_amplitudes**2) - 0.5
            loss = torch.sum(loss)
            return loss

        def total_loss_func(y_pred, y_true):
            loss = basic_loss_func(y_pred, y_true)
            # loss += 0.1*frequency_aggregation_loss_function(3) # 频率汇聚
            # loss += 0.3 * frequency_aggregation_loss_function(5)  # 频率汇聚
            loss += 0.3 * frequency_aggregation_loss_function(7)  # 频率汇聚
            # loss += 0.2 * frequency_aggregation_loss_function(25)  # 频率汇聚
            # loss += amplitudes_to_less()
            return loss

        return total_loss_func

    def train(self, epochs=1000, batch_size=256,lr=0.001,save_model_path=None):
        frequency_train_data = self.frequency_train_data_culculation(self.train_data)
        frequencies_id = torch.arange(self.frequencies_num).long()
        train_dataset = TensorDataset(frequencies_id.unsqueeze(1), frequency_train_data)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_func = self.regularization_loss_func(torch.nn.MSELoss())
        for epoch in range(epochs + 1):
            super().train()
            # 在频域上训练
            for frequencies_id, frequency_labels in dataloader:
                optimizer.zero_grad()
                frequency_pred = self.model_output_FREQUENCY(frequencies_id)
                # print(frequency_pred.shape)
                # print(frequency_labels.shape)
                loss = loss_func(frequency_pred, frequency_labels)
                # print(loss.item())
                loss.backward()
                optimizer.step()
            # 在时域上训练
            # train_t = torch.arange(0,train_data.shape[0],dtype=torch.float64,device=self.device).unsqueeze(1)
            # time_pred = self.model_output_TIME(train_t,window_size).squeeze(1)
            # loss = loss_func(time_pred,train_data)
            # loss.backward()
            # optimizer.step()

            if epoch % 500 == 0:  # 保持振幅为正数，相位在[0,2π]范围内
                with torch.no_grad():
                    self.trainable_phases[self.trainable_amplitudes < 0] += pi
                    self.trainable_phases.remainder_( 2 * pi) # 取余
                    torch.abs_(self.trainable_amplitudes)

            if epoch % PRINT_EPOCHS == 0:
                if save_model_path is not None:
                    torch.save(self.state_dict(), save_model_path)
                else:
                    torch.save(self.state_dict(), os.path.join(MODEL_DIR, TIME_ID + ".pth"))
                print("epoch:", epoch, "loss:", loss.item())
                self.draw(epoch,None,frequency_train_data)

    def draw(self, epoch, time_train_data=None, frequency_train_data=None):
        '''
        time_train_data和frequency_train_data只接收一条序列，比如请输入self.train_data[0]
        epoch = None 表示直接绘制图像并plt.show()，而不是保存为文件
        '''
        
        time_train_data = self.train_data
        if(frequency_train_data is None):
            frequency_train_data = self.frequency_train_data_culculation(time_train_data)
        
        TIME_FIELD = True  # 时域图
        FREQUENCY_FIELD_SIN = True  # 频域图(sin)
        FREQUENCY_FIELD_COS = True  # 频域图(cos)
        MODEL_AMPLITUDES = True  # 模型振幅
        MODEL_PHASES = 1  # 模型相位
        FFT_RESULT = True  # 快速傅里叶变换得到的频谱图

        imgs_num = (
            TIME_FIELD
            + FREQUENCY_FIELD_SIN
            + FREQUENCY_FIELD_COS
            + MODEL_AMPLITUDES
            + MODEL_PHASES
        )
        imgs_count = 1
        with torch.no_grad():
            plt.figure(figsize=(25, 5 * imgs_num))
            frequencies = self.frequencies.cpu().numpy()
            frequencies_id = torch.arange(self.frequencies_num).long()
            frequency_pred = self.model_output_FREQUENCY(frequencies_id).cpu().numpy()
            frequency_train_data = frequency_train_data.cpu().numpy()
            trainable_amplitudes = self.trainable_amplitudes.detach().cpu().numpy()
            trainable_phases = self.trainable_phases.detach().cpu().numpy()
            if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                fft_pred = torch.fft.rfft(time_train_data)
                fft_amplitudes = torch.abs(fft_pred)*2 / self.sequence_len
                fft_amplitudes = fft_amplitudes.cpu().numpy()
                frequencies_fft = np.arange(0, fft_amplitudes.shape[0])/self.sequence_len
            if TIME_FIELD:  # 时域
                train_signal = (time_train_data + self.mean).cpu().numpy()

                t = torch.arange(
                    -self.sequence_len, 2*self.sequence_len, dtype=torch.float64, device=self.device
                ).unsqueeze(1)  # 测试时用的时间序列
                # t = torch.arange(0,data_len,dtype=torch.float64,device=self.device).unsqueeze(1) # 测试时用的时间序列
                time_pred = self.model_output_TIME(t).cpu().numpy()
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                t = np.arange(-self.sequence_len, 2*self.sequence_len)
                # t = np.arange(0,data_len)
                # y=0横坐标轴
                if IDEAL_SIGNAL:
                    true_signal = func2sequence(signal_func,-self.sequence_len, 2*self.sequence_len)  # 理想真实信号
                    plt.plot(t, true_signal, label="true signal", color="green")
                plt.plot(np.arange(self.sequence_len), train_signal, label="train data", color="blue")
                plt.plot(t, time_pred, label="predict", alpha=0.5, color="red")
                plt.grid(True)
                plt.legend()
            if FREQUENCY_FIELD_SIN:  # 频域sin
                sin_pred = frequency_pred[:, 0]
                sin_data = frequency_train_data[:, 0]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(frequencies, sin_data, label="sin_data", color="green")
                plt.plot(frequencies, sin_pred, label="sin_pred", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend()
            if FREQUENCY_FIELD_COS:  # 频域cos
                cos_pred = frequency_pred[:, 1]  # cos的频域
                cos_data = frequency_train_data[:, 1]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(frequencies, cos_data, label="cos_data", color="green")
                plt.plot(frequencies, cos_pred, label="cos_pred", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend()

            if MODEL_AMPLITUDES:  # 模型振幅
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    plt.scatter(frequencies_fft, fft_amplitudes, label="fft_amplitudes", color="blue", marker="s", s=20)
                    plt.vlines(frequencies_fft, ymin=0, ymax=fft_amplitudes, colors="blue", linestyles="solid")

                plt.scatter(frequencies,trainable_amplitudes,label="model_amplitudes",color="red",marker=".", s=50,alpha=0.7)
                plt.vlines(frequencies,ymin=0,ymax=trainable_amplitudes,colors="red",linestyles="dotted",alpha=0.4)
                if IDEAL_SIGNAL:
                    plt.scatter(1/test_T,test_amplitudes,color="green",marker="*",s=50)
                plt.grid(True)
                plt.legend()

            if MODEL_PHASES:  # 模型相位
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.axhline(y=2*pi, color='gray', linestyle='--')  # y=2π参考线
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    fft_phases = torch.angle(fft_pred)
                    fft_phases = torch.remainder(fft_phases, 2 * pi)
                    alpha = fft_amplitudes**2  # 振幅越小，不透明度越低
                    alpha = alpha / np.mean(alpha) * 0.8
                    alpha[alpha > 0.8] = 0.8
                    fft_phases = fft_phases.cpu().numpy()
                    plt.scatter(frequencies_fft, fft_phases, label="fft_phases", color="blue", marker="s", s=20,alpha=alpha)
                    plt.vlines(frequencies_fft, ymin=0, ymax=fft_phases, colors="blue", linestyles="solid",alpha=alpha)
                
                alpha = trainable_amplitudes**2  # 振幅越小，不透明度越低
                alpha = alpha / np.mean(alpha) * 0.8
                alpha[alpha > 0.8] = 0.8
                plt.scatter(frequencies,trainable_phases,label="model_phases",color="red",marker=".", s=50,alpha=alpha)
                plt.vlines(frequencies,ymin=0,ymax=trainable_phases,colors="red",linestyles="dotted",alpha=alpha)
                
                if IDEAL_SIGNAL:
                    plt.scatter(1/test_T,test_phases,color="green",marker="*",s=50)
                plt.grid(True)
                plt.legend()
        if epoch is not None:
            plt.savefig(os.path.join(IMAGE_DIR, f"epoch{epoch}.png"),bbox_inches="tight")
            plt.close()
        else:
            plt.show()



def signal_func(t):
    T = test_T
    amplitudes = test_amplitudes
    phases = test_phases
    return np.sum(amplitudes * np.sin(2 * np.pi * t / T + phases), axis=0)


def func2sequence(func, begin, end, noisy=0):
    sequence_data = []
    for t in range(begin, end):
        sequence_data.append(func(t))
    sequence_data = np.array(sequence_data)
    if noisy:
        sequence_data += np.random.normal(0, noisy, sequence_data.shape)
    return sequence_data


# 下面是测试用的理想信号
IDEAL_SIGNAL = 0  # 启用理想信号（画图用）
IDEAL_SIGNAL = 1  # 启用理想信号（画图用）
test_T =   1500 / np.array([2, 5, 10, 45, 50,230,240,250,567,570,722,725])  # 周期
test_amplitudes = np.array([4, 2,  4,  2,  5,  2,  4,  2,  3,  4,   1,   2], dtype=np.float64)  # 振幅
test_amplitudes /= np.max(test_amplitudes) / 2  # 归一化
test_phases =     np.array([1, 0,  2,  1,1.2,0.5,3.2,0.3,2.5,  3, 4.2, 3.4])  # 相位
noisy = 0

train_data = func2sequence(signal_func,0,1700,noisy=noisy)


# model = FDLM(train_data)
model = FDLM(train_data,load_model_path=r"models\20250417_224627.pth")
# model.train(epochs=10000,batch_size=256,lr=0.000001)
model.draw(epoch=None)
exit()
