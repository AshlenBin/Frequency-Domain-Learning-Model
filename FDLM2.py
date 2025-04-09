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
频率分布改用快速傅里叶变换的分布形式，即周期分量是序列长度的1/k倍
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
        max_period=None,
        included_times=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        load_model_path=None
    ):
        """
        train_data: 时域上的训练数据，numpy.ndarray或List格式
            train_data的形状应该像这样：
                [[1,2,3,1,2,3,1,2,3],
                 [2,4,2,2,4,2,2,4,2],
                 [1,4,3,1,4,3,1,4,3]]
            即每一行是一条时间序列，每条时间序列相互独立，这样设计只为了并行优化提升效率
            若输入train_data=[1,2,3,1,2,3,1,2,3]则自动转为[[1,2,3,1,2,3,1,2,3]]
        max_period: 最大周期，默认为训练数据长度【待修改】
        included_times: 指定包含的频率分量的周期，在指定范围以外的频率直接被排除【待修改】
        load_model_path: 从路径加载模型，默认为None，也就是从头开始训练
        """
        
        super().__init__()
        self.device = device
        train_data = torch.tensor(train_data, dtype=torch.float64, device=device)
        if(train_data.dim()>2):
            raise ValueError("train_data的纬度应当为1或2")
        if(train_data.dim()==1):
            train_data = train_data.unsqueeze(0)
        self.sequences_num = train_data.shape[0]
        self.sequence_len = train_data.shape[1]
        
        # 对train_data消去常数项
        self.mean = torch.mean(train_data,1,keepdim=True)
        self.train_data = train_data - self.mean  # 给模型的训练数据在用户输入的数据基础上去掉均值，在输出时再加上

        if included_times is not None:
            self.max_period = max(included_times)
        else:
            self.max_period = self.sequence_len//2 if max_period is None else max_period
        
        self.Kc = torch.arange(1, self.sequence_len//2+1,dtype=torch.long)
        self.periods = self.sequence_len / self.Kc.type(torch.float64).to(self.device) # 常数：周期

        self.trainable_amplitudes = nn.Parameter(
            torch.ones((self.sequences_num,len(self.Kc)), device=device, dtype=torch.float64)
        )  # 可训练参数：振幅
        self.trainable_phases = nn.Parameter(
            torch.ones((self.sequences_num,len(self.Kc)), device=device, dtype=torch.float64)
        )  # 可训练参数：相位

        self.frequency_convolution_coefficients_A = None  # 频率卷积系数
        self.frequency_convolution_coefficients_B = None  # 频率卷积系数
        included_times = np.array(included_times) if included_times is not None else None
        self.frequency_model_coefficients_culculation(included_times)  # 计算频率卷积系数

        # self.frequency_train_data = None
        # self.frequency_train_data_culculation() # 计算实验采样信号的频率域信号
        # self.train_dataset = TensorDataset(self.periods.long().unsqueeze(1),self.frequency_train_data)
        
        if load_model_path is not None:
            self.load_state_dict(torch.load(load_model_path))
            print("load model from: ", load_model_path)

    def frequency_train_data_culculation(self, time_train_data):
        # print(ground_truth)
        N = self.sequence_len
        # 假设N=1000，那么矩阵大小就是N*N=10^6，2GB显存可以容纳2^28=268,435,456个torch.float64，是够用的
        # 若N超过10000则将它拆分为两半分别计算再相加(未实现)
        Tc = self.periods.reshape(-1, 1)
        n = torch.arange(N, dtype=torch.float64, device=self.device).unsqueeze(0)
        __matrix = (2 * pi / Tc) @ n
        time_train_data = time_train_data.transpose(0, 1)
        sin_sum = sin(__matrix) @ time_train_data
        cos_sum = cos(__matrix) @ time_train_data
        # sin_sum[0,:] = 0 # 【特别地】当Tc=2时，sin总和为0

        return torch.stack((sin_sum, cos_sum), dim=2)
        # self.frequency_train_data的维度是：(Kc,sequences_num,sin和cos)

    def frequency_model_coefficients_culculation(self, included_times):
        N = self.sequence_len
        T = self.periods
        # 一定要把T转为float64类型，否则会出现精度问题。不能在alpha之后才转精度！
        alpha = pi / T
        alpha = alpha.repeat(len(T), 1)
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

        if included_times is not None:  #
            excluded_times = np.setdiff1d(np.arange(2, self.max_period + 1), included_times)
            A[:, :, excluded_times - 2] = 0

        self.frequency_convolution_coefficients_A = A
        self.frequency_convolution_coefficients_B = B
        # self.frequency_convolution_coefficients_A的维度是：(sin和cos, Tc, T)
        # 它与trainable_amplitudes做矩乘后frequency_output的维度是：(sin和cos, Tc, sequences_num)
        # 而frequency_train_data的维度是：(Tc, sin和cos（不分开）)
        # 所以记得要做维度转换：frequency_output.transpose(0,1).squeeze(2)
        # plt.plot(self.frequency_convolution_coefficients_A[0,5, :].cpu().numpy())
        # plt.show()

    def __call__(self, t: torch.Tensor):
        # 前向函数仅用于在应用时输出预测值，不用于训练
        # 目前暂时只接受单变量输入，比如[[1],[2],[3]]，表示要分别获取n=1,n=2,n=3的预测值
        if t.dim() == 1:  # 如果输入[1,2,3]，自动转为[[1],[2],[3]]
            t = t.unsqueeze(1)

        t = t.to(self.device)
        with torch.no_grad():
            t = self.model_output_TIME(t)
        return t

    def model_output_FREQUENCY(self, Kc: torch.Tensor):
        # 频域上的模型输出，用于训练
        Kc = Kc.squeeze(1)
        # print(Tc)
        coefficients_A = self.frequency_convolution_coefficients_A[:, Kc-1]
        coefficients_B = self.frequency_convolution_coefficients_B[:, Kc-1]
        # print(coefficients_A.shape)
        # print(coefficients_B.shape)
        # print(self.trainable_phases.shape)
        cB_add_phases = coefficients_B.repeat(self.sequences_num,1,1,1) + self.trainable_phases.unsqueeze(1).unsqueeze(1)
        # print('------------')
        # print(coefficients_B)
        # print(cB_add_phases)
        # exit()
        frequency_output = coefficients_A * sin(cB_add_phases)

        frequency_output = frequency_output @ self.trainable_amplitudes.unsqueeze(1).unsqueeze(3)

        return frequency_output.transpose(1, 2).squeeze(3)

    def model_output_TIME(self, t: torch.Tensor):
        T = self.periods

        t = t * 2 * torch.pi
        t = t.repeat(1, len(T))/T
        t = t + self.trainable_phases.unsqueeze(1)

        t = torch.sin(t) @ self.trainable_amplitudes.unsqueeze(2)
        return t.squeeze(2) + self.mean

    def regularization_loss_func(self, basic_loss_func=torch.nn.MSELoss()):

        relu = torch.functional.F.relu

        def frequency_aggregation_loss_function(SCALE):
            # 频率汇聚损失函数，使得频率强度尽可能集中到单个频率上
            kernel = torch.arange(1, SCALE + 1, dtype=torch.float64, device=self.device)  # SCALE=5 则 kernel=[1,2,3,4,5]
            A = self.trainable_amplitudes
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
            A = self.trainable_amplitudes
            A = A / torch.abs(A.detach()).mean(dim=1,keepdim=True)
            loss = F.sigmoid(2*(A**2)) - 0.5
            loss = torch.sum(loss)
            return loss

        def total_loss_func(y_pred, y_true):
            loss = basic_loss_func(y_pred, y_true)
            # loss += 0.1*frequency_aggregation_loss_function(3) # 频率汇聚
            # loss += 0.3 * frequency_aggregation_loss_function(5)  # 频率汇聚
            # loss += 0.3 * frequency_aggregation_loss_function(13)  # 频率汇聚
            # loss += 0.2 * frequency_aggregation_loss_function(25)  # 频率汇聚
            loss += amplitudes_to_less()
            return loss

        return total_loss_func

    def train(self, epochs=5000, batch_size=256,lr=0.0001,save_model_path=None):
        frequency_train_data = self.frequency_train_data_culculation(self.train_data)
        Kc = self.Kc.unsqueeze(1)
        train_dataset = TensorDataset(Kc.long(), frequency_train_data)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=0.5)  # 不要使用Adam，会导致很多随机零散的频率分量
        loss_func = self.regularization_loss_func(torch.nn.MSELoss())
        for epoch in range(epochs + 1):
            super().train()
            # 在频域上训练
            for Kc, frequency_labels in dataloader:
                optimizer.zero_grad()
                frequency_pred = self.model_output_FREQUENCY(Kc).transpose(0,1)
                # 【注意】self.model_output_FREQUENCY()的输出维度是(sequences_num,Kc,sin和cos)
                # 而frequency_labels的维度是(Kc,sequences_num,sin和cos)，（为了分batch）
                # 所以需要做维度转换：frequency_pred.transpose(0,1).squeeze(2)
                loss = loss_func(frequency_pred, frequency_labels)
                loss.backward()
                optimizer.step()
                
            # 在时域上训练
            # train_t = torch.arange(0,self.sequence_len,dtype=torch.float64,device=self.device).unsqueeze(1)
            # time_pred = self.model_output_TIME(train_t)
            # loss = loss_func(time_pred,self.train_data)
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
                self.draw(epoch,None,frequency_train_data[:,0])

    def draw(self, epoch, time_train_data=None, frequency_train_data=None):
        '''
        time_train_data和frequency_train_data只接收一条序列，比如请输入self.train_data[0]
        epoch = None 表示直接绘制图像并plt.show()，而不是保存为文件
        '''
        if(time_train_data is None):
            time_train_data = self.train_data[0]
        if(frequency_train_data is None):
            frequency_train_data = self.frequency_train_data_culculation(time_train_data.unsqueeze(0))[:,0]
        
        TIME_FIELD = True  # 时域图
        FREQUENCY_FIELD_SIN = True  # 频域图(sin)
        FREQUENCY_FIELD_COS = True  # 频域图(cos)
        MODEL_AMPLITUDES = True  # 模型振幅
        MODEL_PHASES = 1  # 模型相位
        FFT_RESULT = True  # 快速傅里叶变换得到的频谱图
        X_AXIS_T = 0  # 横坐标为 周期(True) 或 序列长度的N/x倍(False)

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
            Kc = self.Kc.unsqueeze(1)
            frequency_pred = self.model_output_FREQUENCY(Kc)[0].cpu().numpy()
            frequency_train_data = frequency_train_data.cpu().numpy()
            trainable_amplitudes = self.trainable_amplitudes[0].cpu().numpy()
            trainable_phases = self.trainable_phases[0].cpu().numpy()
            if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                fft_pred = torch.fft.rfft(time_train_data)
            if X_AXIS_T:
                Kc = self.sequence_len / self.periods.cpu().numpy()
            if TIME_FIELD:  # 时域
                train_signal = (time_train_data + self.mean)[0].cpu().numpy()

                t = torch.arange(
                    -self.sequence_len, 2*self.sequence_len, dtype=torch.float64, device=self.device
                ).unsqueeze(1)  # 测试时用的时间序列
                # t = torch.arange(0,data_len,dtype=torch.float64,device=self.device).unsqueeze(1) # 测试时用的时间序列
                time_pred = self.model_output_TIME(t)[0].cpu().numpy()
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
                plt.plot(Kc, sin_data, label="sin_data", color="green")
                plt.plot(Kc, sin_pred, label="sin_pred", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend()
            if FREQUENCY_FIELD_COS:  # 频域cos
                cos_pred = frequency_pred[:, 1]  # cos的频域
                cos_data = frequency_train_data[:, 1]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(Kc, cos_data, label="cos_data", color="green")
                plt.plot(Kc, cos_pred, label="cos_pred", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend()

            if MODEL_AMPLITUDES:  # 模型振幅
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    fft_amplitudes = torch.abs(fft_pred)*2 / self.sequence_len
                    x = np.arange(0, fft_amplitudes.shape[0])
                    if X_AXIS_T:
                        x[0] = self.sequence_len
                        x = self.sequence_len / x
                        fft_amplitudes = fft_amplitudes[x <= self.max_period]
                        x = x[x <= self.max_period]
                    fft_amplitudes = fft_amplitudes.cpu().numpy()
                    plt.scatter(x, fft_amplitudes, label="fft_amplitudes", color="blue", marker=".", s=50)
                    plt.vlines(x, ymin=0, ymax=fft_amplitudes, colors="blue", linestyles="dotted")

                plt.scatter(Kc,trainable_amplitudes,label="model_amplitudes",color="red",marker=".", s=50,alpha=0.7)
                plt.vlines(Kc,ymin=0,ymax=trainable_amplitudes,colors="red",linestyles="solid",alpha=0.4)
                if IDEAL_SIGNAL:
                    if X_AXIS_T:
                        plt.scatter(test_T, test_amplitudes, color="green", marker="*", s=50)
                    else:
                        plt.scatter(self.sequence_len / test_T,test_amplitudes,color="green",marker="*",s=50)
                plt.grid(True)
                plt.legend()

            if MODEL_PHASES:  # 模型相位
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    fft_phases = torch.angle(fft_pred)
                    fft_phases = torch.remainder(fft_phases, 2 * pi)
                    x = np.arange(0, fft_phases.shape[0])
                    if X_AXIS_T:
                        x[0] = self.sequence_len
                        x = self.sequence_len / x
                        fft_phases = fft_phases[x <= self.max_period]
                        x = x[x <= self.max_period]
                    fft_phases = fft_phases.cpu().numpy()
                    plt.scatter(x, fft_phases, label="fft_phases", color="blue", marker=".", s=50)
                    plt.vlines(x, ymin=0, ymax=fft_phases, colors="blue", linestyles="dotted")
                
                alpha = (trainable_amplitudes)**2  # 振幅越小，不透明度越低
                alpha = alpha / np.mean(alpha) * 0.8
                alpha[alpha > 0.8] = 0.8
                plt.scatter(Kc,trainable_phases,label="model_phases",color="red",marker=".", s=50,alpha=alpha)
                plt.vlines(Kc,ymin=0,ymax=trainable_phases,colors="red",linestyles="solid",alpha=alpha)
                
                if IDEAL_SIGNAL:
                    if X_AXIS_T:
                        plt.scatter(test_T, test_phases, color="green", marker="*", s=50)
                    else:
                        plt.scatter(self.sequence_len / test_T,test_phases,color="green",marker="*",s=50)
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

train_data = func2sequence(signal_func,0,1500,noisy=noisy)

model = FDLM(train_data)
model.train(epochs=10000,batch_size=32,lr=0.00001)
model.draw(epoch=None)
exit()


from datasets import export_mako

# train_data = export_mako.export("mako6", 10, 10)
# train_data = torch.tensor(train_data, device="cuda")
# train_data = train_data / torch.std(train_data) / 2

# model = FDLM(train_data, max_period=2000,load_model_path=r"models\20250408_164415.pth")
# model.draw(epoch=None)
# model.train(epochs=20000, batch_size=1024)

train_data = export_mako.export_retangle("mako6", [10, 30], [10,30])
train_data = torch.tensor(train_data)
train_data = train_data / torch.std(train_data) / 2

begin_time = time.time()
print("start training...")
path_dict = [f"mako6/mako6_(x=[10~30],y=[10~15]).pth",
             f"mako6/mako6_(x=[10~30],y=[15~20]).pth",
             f"mako6/mako6_(x=[10~30],y=[20~25]).pth",
             f"mako6/mako6_(x=[10~30],y=[25~30]).pth"]

# model = FDLM(train_data[0:5,:].reshape(-1,train_data.shape[2]))
# # model.train(epochs=20000, batch_size=256,save_model_path=MODEL_DIR + f"mako6/mako6_(x=[10~30],y=[10~15]).pth")
# model = FDLM(train_data[5:10,:].reshape(-1,train_data.shape[2]))
# # model.train(epochs=20000, batch_size=256,save_model_path=MODEL_DIR + f"mako6/mako6_(x=[10~30],y=[15~20]).pth")
# model = FDLM(train_data[10:15,:].reshape(-1,train_data.shape[2]))
# # model.train(epochs=20000, batch_size=256,save_model_path=MODEL_DIR + f"mako6/mako6_(x=[10~30],y=[20~25]).pth")
# model = FDLM(train_data[15:20,:].reshape(-1,train_data.shape[2]))
# model.train(epochs=20000, batch_size=256,save_model_path=MODEL_DIR + f"mako6/mako6_(x=[10~30],y=[25~30]).pth")
for i in range(20):
    model = FDLM(train_data[i])
    model.train(epochs=10000, batch_size=256,save_model_path=MODEL_DIR + f"mako6/mako6_(y={i+10},x=[10~30]).pth")


time_cost = time.time() - begin_time
print("time cost:", time_cost)

# model = FDLM(train_data[0:1,:].reshape(-1,train_data.shape[2])
#              ,load_model_path=MODEL_DIR + path_dict[0])
# print(train_data[0:1,:].reshape(-1,train_data.shape[2]).shape)
