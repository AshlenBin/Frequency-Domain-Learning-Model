import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import sin, cos, atan2, sqrt, pi
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import time
import os

"""
把频率w也作为可训练参数
直接使用傅立叶变换来初始化参数
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
    a_time = 0
    b_time = 0
    c_time = 0
    def __init__(
        self,
        train_data,
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
        self.trainable_k = nn.Parameter(
            torch.arange(1,self.sequence_len//2+1,dtype=torch.float64,device=device)
        )  # 可训练参数：频率分量
        # 直接使用傅立叶变换来初始化参数
        fft_pred = torch.fft.rfft(self.train_data)
        fft_amplitudes = torch.abs(fft_pred)*2 / self.sequence_len
        fft_phases = torch.angle(fft_pred).remainder_( 2 * pi)
        self.trainable_amplitudes = nn.Parameter(fft_amplitudes[1:].type(torch.float64).to(self.device))  # 可训练参数：振幅
        self.trainable_phases = nn.Parameter(fft_phases[1:].type(torch.float64).to(self.device))  # 可训练参数：相位
        
        if load_model_path is not None:
            self.load_state_dict(torch.load(load_model_path))
            print("load model from: ", load_model_path)

    def frequency_train_data_culculation(self, time_train_data):
        fft_result = torch.fft.rfft(time_train_data)
        real = torch.real(fft_result)[1:]
        imag = torch.imag(fft_result)[1:]
        result = torch.stack((real, imag), dim=1).type(torch.float64)
        return result
        # frequency_train_data的维度是：
        #   [[s2,c2],
        #    [s3,c3],
        #     ..... ,
        #    [sN,cN]]

    def __call__(self, t: torch.Tensor):
        # 前向函数仅用于在应用时输出预测值，不用于训练
        # 目前暂时只接受单变量输入，比如[[1],[2],[3]]，表示要分别获取n=1,n=2,n=3的预测值
        if t.dim() == 1:  # 如果输入[1,2,3]，自动转为[[1],[2],[3]]
            t = t.unsqueeze(1)

        t = t.to(self.device)
        with torch.no_grad():
            t = self.model_output_TIME(t)
        return t

    def model_output_FREQUENCY(self,kc:torch.Tensor):
        # 频域上的模型输出，用于训练
        # 【注意】frequencies_id是频率值在self.frequencies中的下标，不是频率值本身
        # 【规定】为了优化计算速度，对于kc，只取其左右±10的k值参与叠加，因为再远一些频谱泄漏已经很小了
        if(kc.dim() == 1):  # 如果输入[1,2,3]，自动转为[[1],[2],[3]]
            kc = kc.unsqueeze(1)
        
        
        # start = time.perf_counter()
        
        N = self.sequence_len
        kc = kc.repeat(1,len(self.trainable_k))
        k = self.trainable_k.repeat(kc.shape[0],1)
        phi = self.trainable_phases.repeat(kc.shape[0],1)
        A = self.trainable_amplitudes.repeat(kc.shape[0],1)
        ddddd = torch.abs(k.detach()-kc)
        index = (ddddd < 10)  # 为了优化计算速度，对于kc，只取其左右±10的k值参与叠加，因为再远一些频谱泄漏已经很小了
        # print(index)
        k = k[index]
        kc = kc[index]
        phi = phi[index]
        A = A[index]
        alpha_sub = pi/N*(k-kc)
        alpha_sub[alpha_sub==0] += 1e-9  # 避免除0
        alpha_add = pi/N*(k+kc)
        # 【注意】当alpha_sub=0时ss_sub除0会出现nan值，你如果直接用N覆盖可以保证前向传播输出不出错，但是反向传播求梯度的时候依然会出错！
        ss_add = sin(N*alpha_add)/sin(alpha_add)/2
        ss_sub = sin(N*alpha_sub)/sin(alpha_sub)/2
        real = (
            cos((N-1)*alpha_add+phi)*ss_add
            + cos((N-1)*alpha_sub+phi)*ss_sub
        )
        imag = (
            sin((N-1)*alpha_sub+phi)*ss_sub
            - sin((N-1)*alpha_add+phi)*ss_add
        )
        real_imag = torch.stack((real,imag),dim=1)* A.unsqueeze(1)
        real_imag = real_imag
        
        # self.a_time += time.perf_counter() - start
        # start = time.perf_counter()
        
        index = index.sum(dim=1)
        index = torch.cumsum(index.cpu(),dim=0)
        real_imag = torch.tensor_split(real_imag,index[:-1])
        real_imag = torch.nn.utils.rnn.pad_sequence(real_imag, batch_first=True)
        real_imag = real_imag.sum(dim=1)
        # exit()
        # self.b_time += time.perf_counter() - start

        return real_imag

    def model_output_TIME(self, t: torch.Tensor):
        # t的维度长这样：[[1],[2],[3]]
        t = t * 2 * torch.pi/self.sequence_len
        t = t.repeat(1, self.trainable_k.shape[0])
        t = t * self.trainable_k + self.trainable_phases
        t = torch.cos(t) @ self.trainable_amplitudes.reshape(-1, 1)
        return t + self.mean

    def regularization_loss_func(self, basic_loss_func=torch.nn.MSELoss()):

        relu = torch.functional.F.relu

        def frequency_aggregation_loss_function(scale):
            # 旁瓣压抑损失函数，在scale区间内，低于μ+σ的振幅值被压低，高于μ+σ的振幅值保持不变
            # scale的值越大，并非使得压抑的范围更大，反而是对偏大值更加宽容，因为扩大范围往往纳入更多偏小值，拉低μ值
            A_data = self.trainable_amplitudes.detach().unfold(0,scale,1)
            A_param = self.trainable_amplitudes.unfold(0,scale,1)
            mean = torch.mean(A_data,dim=1,keepdim=True)
            std = torch.std(A_data,dim=1,keepdim=True)
            threshold = mean+std
            loss = torch.sum(A_param[A_data<=threshold]**2)/scale
            return loss
        
        def keep_k_stay():  # 可训练参数k会到处乱跑，这里把它限制在初始位置的±2以内
            distance = self.trainable_k - torch.arange(1,self.sequence_len//2+1,dtype=torch.float64,device=self.device)
            loss = (relu(distance-2)+relu(-distance-2))**2 * 10
            return loss.sum()

        def amplitudes_to_less():  # 让接近0的值直接跑向0
            loss = F.sigmoid(self.trainable_amplitudes**2) - 0.5
            loss = torch.sum(loss)
            return loss

        def total_loss_func(y_pred, y_true):
            loss = basic_loss_func(y_pred, y_true)
            loss += 10 * frequency_aggregation_loss_function(9)  # 旁瓣压抑
            loss += 5 * amplitudes_to_less()
            loss += keep_k_stay()
            return loss

        return total_loss_func

    def train(self, epochs=1000, batch_size=256,lr=0.001,save_model_path=None):
        frequency_train_data = self.frequency_train_data_culculation(self.train_data)
        kc = torch.arange(1,self.sequence_len//2 +1)
        train_dataset = TensorDataset(kc.unsqueeze(1), frequency_train_data)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = self.regularization_loss_func(torch.nn.MSELoss())
        for epoch in range(epochs + 1):
            super().train()
            # 在频域上训练
            for kc, frequency_labels in dataloader:
                kc = kc.to(self.device)
                frequency_labels = frequency_labels.to(self.device)
                optimizer.zero_grad()
                frequency_pred = self.model_output_FREQUENCY(kc)
                # print(frequency_pred.shape)
                # print(frequency_labels.shape)
                # start = time.perf_counter()
                loss = loss_func(frequency_pred, frequency_labels)
                # self.c_time += time.perf_counter() - start
                # print(loss.item())
                # print("a_time:",self.a_time,"b_time:",self.b_time,"c_time:",self.c_time)
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
        FREQUENCY_FIELD_REAL = True  # 频域图(实部)
        FREQUENCY_FIELD_IMAG = True  # 频域图(虚部)
        MODEL_AMPLITUDES = True  # 模型振幅
        MODEL_PHASES = 1  # 模型相位
        FFT_RESULT = True  # 快速傅里叶变换得到的频谱图

        imgs_num = (
            TIME_FIELD
            + FREQUENCY_FIELD_REAL
            + FREQUENCY_FIELD_IMAG
            + MODEL_AMPLITUDES
            + MODEL_PHASES
        )
        imgs_count = 1
        with torch.no_grad():
            plt.figure(figsize=(25, 5 * imgs_num))
            kc = torch.arange(1,self.sequence_len//2 +1)
            frequency_pred = self.model_output_FREQUENCY(kc.to(self.device)).cpu().numpy()
            k_pred = self.trainable_k.detach().cpu().numpy()
            k_label = np.arange(1,self.sequence_len//2 +1)
            frequency_train_data = frequency_train_data.cpu().numpy()
            trainable_amplitudes = self.trainable_amplitudes.detach().cpu().numpy()
            trainable_phases = self.trainable_phases.detach().cpu().numpy()
            if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                fft_pred = torch.fft.rfft(time_train_data)
                fft_amplitudes = torch.abs(fft_pred)*2 / self.sequence_len
                fft_amplitudes = fft_amplitudes.cpu().numpy()
                k_fft = np.arange(0, fft_amplitudes.shape[0])
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
                plt.legend(loc='upper right')
            if FREQUENCY_FIELD_REAL:  # 频域图(实部)
                real_pred = frequency_pred[:, 0]
                real_data = frequency_train_data[:, 0]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(k_label, real_data, label="real_data", color="green")
                plt.plot(k_label, real_pred, label="real_pred", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend(loc='upper right')
            if FREQUENCY_FIELD_IMAG:  # 频域图(虚部)
                cos_pred = frequency_pred[:, 1]  # cos的频域
                imag_data = frequency_train_data[:, 1]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(k_label, imag_data, label="imag_data", color="green")
                plt.plot(k_label, cos_pred, label="cos_pred", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend(loc='upper right')

            if MODEL_AMPLITUDES:  # 模型振幅
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    plt.scatter(k_fft, fft_amplitudes, label="fft_amplitudes", color="blue", marker="s", s=2,alpha=0.7)
                    plt.vlines(k_fft, ymin=0, ymax=fft_amplitudes, colors="blue", linestyles="solid",alpha=0.7)
                plt.scatter(k_pred,trainable_amplitudes,label="model_amplitudes",color="red",marker=".", s=10,alpha=0.7)
                plt.vlines(k_pred,ymin=0,ymax=trainable_amplitudes,colors="red",linestyles="--",alpha=0.7)
                if IDEAL_SIGNAL:
                    plt.scatter(self.sequence_len / test_T,test_amplitudes,label="true_phases",color="green",marker="*",s=10)
                plt.grid(True)
                plt.legend(loc='upper right')

            if MODEL_PHASES:  # 模型相位
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.axhline(y=2*pi, color='gray', linestyle='--')  # y=2π参考线
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    fft_phases = torch.angle(fft_pred)
                    fft_phases = torch.remainder(fft_phases, 2 * pi)
                    alpha = fft_amplitudes**2  # 振幅越小，不透明度越低
                    alpha = alpha / np.mean(alpha[alpha>np.mean(alpha)]) * 0.8
                    alpha[alpha > 0.8] = 0.8
                    fft_phases = fft_phases.cpu().numpy()
                    plt.scatter(k_fft, fft_phases, color="blue", marker="s", s=5,alpha=alpha)
                    plt.scatter([], [], label="fft_phases", color="blue", marker="s", s=5,alpha=1) # 用于指定图例样式，因为不透明度不一样
                    plt.vlines(k_fft, ymin=0, ymax=fft_phases, colors="blue", linestyles="solid",alpha=alpha)
                
                alpha = trainable_amplitudes**2  # 振幅越小，不透明度越低
                alpha = alpha / np.mean(alpha[alpha>np.mean(alpha)]) * 0.8
                alpha[alpha > 0.8] = 0.8
                plt.scatter(k_pred,trainable_phases,color="red",marker=".", s=15,alpha=alpha)
                plt.scatter([], [], label="model_phases", color="red",marker=".", s=15, alpha=1) # 用于指定图例样式，因为不透明度不一样
                plt.vlines(k_pred,ymin=0,ymax=trainable_phases,colors="red",linestyles="--",alpha=alpha)
                
                if IDEAL_SIGNAL:
                    plt.scatter(self.sequence_len/test_T,test_phases,color="green",marker="*",s=50)
                plt.grid(True)
                plt.legend(loc='upper right')
        if epoch is not None:
            plt.savefig(os.path.join(IMAGE_DIR, f"epoch{epoch}.png"),bbox_inches="tight")
            plt.close()
        else:
            plt.show()



def signal_func(t):
    T = test_T
    amplitudes = test_amplitudes
    phases = test_phases
    return np.sum(amplitudes * np.cos(2 * np.pi * t / T + phases), axis=0)


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
test_T =   1500 / np.array([2, 5, 10, 42, 45, 50,228,230,240,250,567,570,722,725])  # 周期
test_amplitudes = np.array([4, 2,  4,  3,  2,  5,  2,  2,  3,  2,  3,  4,  1,  2], dtype=np.float64)  # 振幅
test_amplitudes /= np.max(test_amplitudes) / 2  # 归一化
test_phases =     np.array([1, 0,  2,  2,  1,1.2,1.2,0.5,3.2,0.3,2.5,  3,4.2,3.4])  # 相位
noisy = 0

train_data = func2sequence(signal_func,0,1700,noisy=noisy)


model = FDLM(train_data)
model.train(epochs=2000,batch_size=256,lr=0.01)
# # model = FDLM(train_data,load_model_path=r"models\20250421_145926.pth")
model.draw(epoch=None)

exit()


from datasets import export_mako

train_data = export_mako.export("mako6", 10, 10)
train_data = torch.tensor(train_data, device="cuda")
train_data = train_data / torch.std(train_data) / 2

model = FDLM(train_data)
# model = FDLM(train_data,load_model_path=r"models\20250421_142825.pth")
model.train(epochs=10000, batch_size=512)
model.draw(epoch=None)
exit()