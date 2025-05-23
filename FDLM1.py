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
    ):
        """
        train_data: 时域上的训练数据，numpy.ndarray或List格式
        max_period: 最大周期，默认为训练数据长度
        included_times: 指定包含的频率分量的周期，在指定范围以外的频率直接被排除
        """
        super().__init__()
        self.device = device
        # 对train_data消去常数项
        train_data = torch.tensor(train_data, dtype=torch.float64, device=device)
        self.mean = torch.mean(train_data)
        self.train_data = (
            train_data - self.mean
        )  # 给模型的训练数据在用户输入的数据基础上去掉均值，在输出时再加上

        if included_times is not None:
            self.max_period = max(included_times)
        else:
            self.max_period = len(train_data) if max_period is None else max_period
        self.periods = torch.arange(
            2, self.max_period + 1, device=device, dtype=torch.float64
        )  # 常数：周期

        self.trainable_amplitudes = nn.Parameter(
            torch.ones(self.max_period - 1, device=device, dtype=torch.float64)
        )  # 可训练参数：振幅
        self.trainable_phases = nn.Parameter(
            torch.randn(self.max_period - 1, device=device, dtype=torch.float64)
            * torch.pi
        )  # 可训练参数：相位

        # fft_result = torch.fft.rfft(self.train_data)
        # self.trainable_amplitudes = nn.Parameter(torch.abs(fft_result).type(torch.float64)) # 可训练参数：振幅
        # self.trainable_phases = nn.Parameter(torch.angle(fft_result).type(torch.float64)) # 可训练参数：相位

        self.frequency_convolution_coefficients_A = None  # 频率卷积系数
        self.frequency_convolution_coefficients_B = None  # 频率卷积系数
        included_times = (
            np.array(included_times) if included_times is not None else None
        )
        self.frequency_model_coefficients_culculation(
            included_times
        )  # 计算频率卷积系数

        # self.frequency_train_data = None
        # self.frequency_train_data_culculation() # 计算实验采样信号的频率域信号
        # self.train_dataset = TensorDataset(self.periods.long().unsqueeze(1),self.frequency_train_data)

    def frequency_train_data_culculation(self, time_train_data):
        # print(ground_truth)
        N = time_train_data.shape[0]
        # 假设N=1000，那么矩阵大小就是N*N=10^6，2GB显存可以容纳2^28=268,435,456个torch.float64，是够用的
        # 若N超过10000则将它拆分为两半分别计算再相加(未实现)
        Tc = self.periods.reshape(-1, 1)
        n = torch.arange(N, dtype=torch.float64, device=self.device).unsqueeze(0)
        __matrix = (2 * pi / Tc) @ n

        time_train_data = time_train_data.reshape(-1, 1)
        sin_sum = sin(__matrix) @ time_train_data
        cos_sum = cos(__matrix) @ time_train_data
        sin_sum[0, :] = 0  # 【特别地】当Tc=2时，sin总和为0
        return torch.cat((sin_sum, cos_sum), dim=1)
        # self.frequency_train_data的维度是：
        #   [[s2,c2],
        #    [s3,c3],
        #     ..... ,
        #    [sN,cN]]

    def frequency_model_coefficients_culculation(self, included_times):
        N = self.train_data.shape[0]
        T = self.periods
        # 一定要把T转为float64类型，否则会出现精度问题。不能在alpha之后才转精度！
        alpha = pi / T
        alpha = alpha.repeat(self.max_period - 1, 1)
        gamma = N * alpha
        beta = gamma - alpha

        alpha1_add_alpha2 = alpha + alpha.T
        alpha1_sub_alpha2 = alpha - alpha.T
        gamma1_add_gamma2 = gamma + gamma.T
        gamma1_sub_gamma2 = gamma - gamma.T

        # 计算K1和K2
        SdS_add = sin(gamma1_add_gamma2) / sin(alpha1_add_alpha2)
        SdS_sub = sin(gamma1_sub_gamma2) / sin(alpha1_sub_alpha2)
        SdS_sub.fill_diagonal_(
            N
        )  # 处理对角线 $T=T_{c}$ 的情况，使得 sin(γ_1 -γ_2 )/sin(α_1 -α_2 ) = N

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
            excluded_times = np.setdiff1d(
                np.arange(2, self.max_period + 1), included_times
            )
            A[:, :, excluded_times - 2] = 0

        self.frequency_convolution_coefficients_A = A
        self.frequency_convolution_coefficients_B = B
        # print(self.frequency_convolution_coefficients_A.shape)
        # self.frequency_convolution_coefficients_A的维度是：(sin和cos, Tc, T)
        # 它与trainable_amplitudes做矩乘后frequency_output的维度是：(sin和cos, Tc, 1)
        # 而frequency_train_data的维度是：(Tc, sin和cos（不分开）)
        # 所以记得要做维度转换：frequency_output.transpose(0,1).squeeze(2)

    def __call__(self, t: torch.Tensor):
        # 前向函数仅用于在应用时输出预测值，不用于训练
        # 目前暂时只接受单变量输入，比如[[1],[2],[3]]，表示要分别获取n=1,n=2,n=3的预测值
        if t.dim() == 1:  # 如果输入[1,2,3]，自动转为[[1],[2],[3]]
            t = t.unsqueeze(1)

        t = t.to(self.device)
        with torch.no_grad():
            t = self.model_output_TIME(t)
        return t

    def model_output_FREQUENCY(self, Tc: torch.Tensor, window_size=1):
        # 频域上的模型输出，用于训练
        Tc = Tc.squeeze(1)
        # print(Tc)
        coefficients_A = self.frequency_convolution_coefficients_A[:, Tc - 2]
        coefficients_B = self.frequency_convolution_coefficients_B[:, Tc - 2]
        # print(coefficients_A.shape)
        # print(coefficients_B.shape)
        # print(self.trainable_phases.shape)
        T = self.periods

        windowed_amplitudes = self.trainable_amplitudes * (
            sin(window_size * pi / T) / (window_size * sin(pi / T))
        )
        windowed_phases = self.trainable_phases + pi * (window_size - 1) / T

        frequency_output = coefficients_A * sin(coefficients_B + windowed_phases)
        frequency_output = frequency_output @ windowed_amplitudes.reshape(-1, 1)
        return frequency_output.transpose(0, 1).squeeze(2)

    def model_output_TIME(self, t: torch.Tensor, window_size=1):
        T = self.periods
        windowed_amplitudes = self.trainable_amplitudes * (
            sin(window_size * pi / T) / (window_size * sin(pi / T))
        )
        windowed_phases = self.trainable_phases + pi * (window_size - 1) / T

        t = t * 2 * torch.pi
        t = t.repeat(1, self.max_period - 1)
        t = t / self.periods + windowed_phases
        t = torch.sin(t) @ windowed_amplitudes.reshape(-1, 1)
        return t + self.mean

    def regularization_loss_func(self, basic_loss_func=torch.nn.MSELoss()):

        relu = torch.functional.F.relu

        def trainable_amplitudes_regularization(trainable_amplitudes):
            # 把可训练参数 $A_{T}$ 的取值范围约束在 $[0,+∞]$
            loss = relu(-trainable_amplitudes + 0.5)
            loss = torch.sum(loss**2)
            # print('amplitudes_regularization_loss:',loss)
            return loss

        def trainable_phases_regularization(trainable_phases):
            # 把可训练参数 $φ_{T}$ 的取值范围约束在 $[-π,π]$
            loss = relu(trainable_phases - pi - 0.1) + relu(
                -trainable_phases + pi + 0.1
            )
            loss = torch.sum(loss**2)
            return loss

        # def frequency_aggregation_loss_function(SCALE):
        #     # 频率汇聚损失函数，使得频率尽可能集中到单个频率上
        #     # 具体操作方式就是计算每5个频率之间的信息熵，使得信息熵尽量小
        #     A = self.trainable_amplitudes.view(1,-1)
        #     A = relu(A)+relu(-A)  # 把负的振幅变成正的
        #     A = torch.log(A+1)
        #     A_sum =  A.unfold(dimension=1, size=SCALE, step=1).sum(dim=2).squeeze(0)
        #     A = A.squeeze(0)
        #     log_sum = torch.log(A_sum)
        #     log_A = torch.log(A)
        #     entropy = torch.zeros_like(A_sum,device=self.device)
        #     # print(amplitudes.shape)
        #     # print(average.shape)
        #     for i in range(0,SCALE):
        #         entropy += (log_A[i:A_sum.shape[0]+i] - log_sum)*A[i:A_sum.shape[0]+i]
        #     entropy = -torch.sum(entropy) # 别忘了取负号
        #     return entropy

        def frequency_aggregation_loss_function(SCALE):
            kernel = torch.arange(
                1, SCALE + 1, dtype=torch.float64, device=self.device
            )  # SCALE=5 则 kernel=[1,2,3,4,5]
            A = self.trainable_amplitudes.view(1, -1)
            A = relu(A) + relu(-A)  # 把负的振幅变成正的
            A_sum = A.unfold(dimension=1, size=SCALE, step=1).sum(dim=2).squeeze(0)
            p = A.unfold(dimension=1, size=SCALE, step=1).transpose(1, 2) / A_sum
            pA_sum = (p.transpose(1, 2) * kernel).sum(dim=2).squeeze(0)
            k = torch.tile(kernel.reshape(-1, 1), (1, pA_sum.shape[0]))
            k = 0.5 * (k - pA_sum) ** 2 + 1
            loss = -torch.sum(p / k)
            return loss

        def amplitudes_to_less():  # 让接近0的值直接跑向0
            loss = F.sigmoid(self.trainable_amplitudes**2) - 0.5
            loss = torch.sum(loss)
            return loss

        def total_loss_func(y_pred, y_true):
            loss = basic_loss_func(y_pred, y_true)
            # loss += trainable_amplitudes_regularization(self.trainable_amplitudes)
            # loss += trainable_phases_regularization(self.trainable_phases)
            # loss += 0.1*frequency_aggregation_loss_function(3) # 频率汇聚
            loss += 0.7 * frequency_aggregation_loss_function(13)  # 频率汇聚
            loss += amplitudes_to_less()
            return loss

        return total_loss_func

    def train(self, epochs=100, batch_size=256):
        frequency_train_data = self.frequency_train_data_culculation(self.train_data)
        train_dataset = TensorDataset(
            self.periods.long().unsqueeze(1), frequency_train_data
        )
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        loss_func = self.regularization_loss_func(torch.nn.MSELoss())
        for epoch in range(epochs + 1):
            super().train()
            # 在频域上训练
            for Tc, frequency_labels in dataloader:
                optimizer.zero_grad()
                frequency_pred = self.model_output_FREQUENCY(Tc)
                loss = loss_func(frequency_pred, frequency_labels)
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
                torch.save(self.state_dict(), os.path.join(MODEL_DIR, TIME_ID + ".pth"))
                print("epoch:", epoch, "loss:", loss.item())
                self.draw(epoch, 1, self.train_data, frequency_train_data)

    def window_train(self, window_size, epochs=100, batch_size=256):
        # 使用方式：
        # model.window_train(16,epochs=2000,batch_size=32)
        # model.window_train(8,epochs=2000,batch_size=32)
        # model.window_train(3,epochs=2000,batch_size=32)
        # model.window_train(2,epochs=2000,batch_size=32)
        # model.window_train(1,epochs=5000,batch_size=32)
        train_data = (
            F.avg_pool1d(
                self.train_data.unsqueeze(0).unsqueeze(0),
                kernel_size=window_size,
                stride=1,
            )
            .squeeze(0)
            .squeeze(0)
        )
        frequency_train_data = self.frequency_train_data_culculation(train_data)
        train_dataset = TensorDataset(
            self.periods.long().unsqueeze(1), frequency_train_data
        )
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        loss_func = self.regularization_loss_func(torch.nn.MSELoss())

        for epoch in range(epochs + 1):
            super().train()
            # 在频域上训练
            for Tc, frequency_labels in dataloader:
                optimizer.zero_grad()
                frequency_pred = self.model_output_FREQUENCY(Tc, window_size)
                loss = loss_func(frequency_pred, frequency_labels)
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
                torch.save(self.state_dict(), os.path.join(MODEL_DIR, TIME_ID + ".pth"))
                print("window_size:", window_size, "epoch:", epoch, "loss:", loss.item())
                with torch.no_grad():
                    self.draw(epoch, window_size, train_data, frequency_train_data)

    def draw(self, epoch, window_size, time_train_data, frequency_train_data):
        TIME_FIELD = True  # 时域图
        FREQUENCY_FIELD_SIN = True  # 频域图(sin)
        FREQUENCY_FIELD_COS = True  # 频域图(cos)
        MODEL_AMPLITUDES = True  # 模型振幅
        MODEL_PHASES = False  # 模型相位
        FFT_RESULT = True  # 快速傅里叶变换得到的频谱图
        X_AXIS_T = 1  # 横坐标为 周期(True) 或 序列长度的N/x倍(False)

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
            data_len = time_train_data.shape[0]
            Tc = torch.arange(2, self.max_period + 1).unsqueeze(1)
            frequency_pred = self.model_output_FREQUENCY(Tc, window_size).cpu().numpy()
            frequency_train_data = frequency_train_data.cpu().numpy()
            if not X_AXIS_T:
                Tc = data_len / self.periods.cpu().numpy()
            if TIME_FIELD:  # 时域
                train_signal = (time_train_data + self.mean).cpu().numpy()

                t = torch.arange(
                    -data_len, 2 * data_len, dtype=torch.float64, device=self.device
                ).unsqueeze(1)  # 测试时用的时间序列
                # t = torch.arange(0,data_len,dtype=torch.float64,device=self.device).unsqueeze(1) # 测试时用的时间序列
                time_pred = self.model_output_TIME(t, window_size).squeeze(1).cpu().numpy()
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                t = np.arange(-data_len, 2 * data_len)
                # t = np.arange(0,data_len)
                # y=0横坐标轴
                if IDEAL_SIGNAL:
                    true_signal = func2sequence(signal_func, -data_len, 2 * data_len + window_size - 1)  # 理想真实信号
                    true_signal = torch.tensor(true_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    true_signal = F.avg_pool1d(true_signal, kernel_size=window_size, stride=1).squeeze(0).squeeze(0).numpy()
                    plt.plot(t, true_signal, label="true signal", color="green")
                plt.plot(np.arange(data_len), train_signal, label="train data", color="blue")
                plt.plot(t, time_pred, label="predict", alpha=0.5, color="red")
                plt.grid(True)
                plt.legend()
            if FREQUENCY_FIELD_SIN:  # 频域sin
                sin_pred = frequency_pred[:, 0]
                sin_data = frequency_train_data[:, 0]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(Tc, sin_data, label="sin_data", color="green")
                plt.plot(Tc, sin_pred, label="sin_pred", color="red")
                plt.grid(True)
                plt.legend()
            if FREQUENCY_FIELD_COS:  # 频域cos
                cos_pred = frequency_pred[:, 1]  # cos的频域
                cos_data = frequency_train_data[:, 1]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(Tc, cos_data, label="cos_data", color="green")
                plt.plot(Tc, cos_pred, label="cos_pred", color="red")
                plt.grid(True)
                plt.legend()

            if MODEL_AMPLITUDES:  # 模型振幅
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    fft_pred = torch.fft.rfft(time_train_data)
                    fft_pred = torch.abs(fft_pred) / data_len
                    x = np.arange(0, fft_pred.shape[0])
                    if X_AXIS_T:
                        x[0] = data_len
                        x = data_len / x
                        fft_pred = fft_pred[x <= self.max_period]
                        x = x[x <= self.max_period]
                    fft_pred = fft_pred.cpu().numpy()
                    plt.scatter(x, fft_pred, label="fft_pred", color="blue", marker=".", s=100)
                    plt.vlines(x, ymin=0, ymax=fft_pred, colors="blue", linestyles="dotted")

                plt.scatter(Tc,self.trainable_amplitudes.cpu().numpy(),label="model_amplitudes",color="red",marker="o",)
                # 绘制垂线
                plt.vlines(Tc,ymin=0,ymax=self.trainable_amplitudes.cpu().numpy(),colors="red",linestyles="dotted",)
                if IDEAL_SIGNAL:
                    if X_AXIS_T:
                        plt.scatter(test_T, test_amplitudes, color="green", marker="*", s=100)
                    else:
                        plt.scatter(data_len / test_T,test_amplitudes,color="green",marker="*",s=100)
                plt.grid(True)
                plt.legend()

            if MODEL_PHASES:  # 模型相位
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(Tc,self.trainable_phases.cpu().numpy(),label="model_phases",color="red",marker="o")
                if IDEAL_SIGNAL:
                    plt.scatter(test_T, test_phases, color="green", marker="*", s=100)
                plt.grid(True)
                plt.legend()

            plt.savefig(os.path.join(IMAGE_DIR, f"window{window_size}_epoch{epoch}.png"),bbox_inches="tight")
            plt.close()


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
test_T = np.array([2, 5, 8, 19, 40, 43, 50])  # 周期
test_amplitudes = np.array([4, 2, 4, 2, 5, 2, 4], dtype=np.float64)  # 振幅
test_amplitudes /= np.max(test_amplitudes) / 2  # 归一化
test_phases = np.array([1, 0, 2, 1, -2, 0, -1])  # 相位
noisy = 0.1
# test_T = np.array([2,5,8,19,40,50])  # 周期
# test_amplitudes = np.array([1,2,3,4,5,2])  # 振幅
# test_phases = np.array([1,0,-2,1,-2,0])  # 相位

# test_T = np.array([5,8,50])  # 周期
# test_amplitudes = np.array([3,4,5])  # 振幅
# test_phases = np.array([1,-2,0])  # 相位

# test_T = np.array([50,49])  # 周期
# test_amplitudes = np.array([5,5])  # 振幅
# test_phases = np.array([-2,0])  # 相位

# train_data = func2sequence(signal_func,0,500,noisy=noisy)

# model = FDLM(train_data,np.max(test_T)+10)
# model.train(epochs=10000,batch_size=32)
# model = FDLM(train_data,60,included_times=[2,5,8,19,40,43,50])
# model.window_train(16,epochs=2000,batch_size=32)
# model.window_train(8,epochs=2000,batch_size=32)
# model.window_train(3,epochs=2000,batch_size=32)
# model.window_train(2,epochs=2000,batch_size=32)
# model.window_train(1,epochs=5000,batch_size=32)


from datasets import export_mako

train_data = export_mako.export("mako6", 10, 10)
train_data = torch.tensor(train_data, device="cuda")
train_data = train_data / torch.std(train_data) / 2

model = FDLM(train_data, max_period=1000)
model.train(epochs=20000, batch_size=1024)
# model.window_train(250,epochs=2000,batch_size=512)
# model.window_train(200,epochs=2000,batch_size=512)
# model.window_train(150,epochs=2000,batch_size=512)
# model.window_train(100,epochs=1000,batch_size=512)
# model.window_train(50,epochs=1000,batch_size=512)
# model.window_train(1,epochs=5000,batch_size=512)
