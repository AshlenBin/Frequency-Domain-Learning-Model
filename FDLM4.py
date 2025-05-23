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
    def __init__(
        self,
        train_data,
        test_data=None,
        test_frequencies_amplitudes_phases=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        load_model_path=None,
    ):
        """
        train_data: 时域上的训练数据，一行，只输入y值不输入t值，t值默认是0,1,2,...
            例：train_data = [1,2,3,1,2,3,1,2,3,1]
        test_data: 时域上的测试数据，两行，第0行t值，第1行y值
            例：test_data = [[-3,-2,-1, 0, 1, 2, 3, 4, 5, 6],
                             [ 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]]
        test_frequencies_amplitudes_phases: 信号的真实频率，幅值，相位
        """
        super().__init__()
        self.device = device
        self.test_data = test_data
        self.test_frequencies_amplitudes_phases = test_frequencies_amplitudes_phases
        # 对train_data消去常数项
        train_data = torch.as_tensor(train_data, dtype=torch.float32, device=device)
        self.mean = torch.mean(train_data)
        self.train_data = train_data - self.mean # 给模型的训练数据在用户输入的数据基础上去掉均值，在输出时再加上
        self.sequence_len = len(train_data)
        
        self.trainable_k = nn.Parameter(
            torch.arange(1,self.sequence_len//2+1,dtype=torch.float32,device=device)
        )  # 可训练参数：频率分量
        # 直接使用傅立叶变换来初始化参数
        fft_pred = torch.fft.rfft(self.train_data)
        fft_amplitudes = torch.abs(fft_pred)*2 / self.sequence_len 
        fft_phases = torch.angle(fft_pred).remainder_( 2 * pi)
        self.trainable_amplitudes = nn.Parameter(fft_amplitudes[1:].type(torch.float32).to(self.device))  # 可训练参数：振幅
        self.trainable_phases = nn.Parameter(fft_phases[1:].type(torch.float32).to(self.device))  # 可训练参数：相位
        
        if load_model_path is not None:
            self.load_model(load_model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        # print("load model from: ", model_path)
    

    def frequency_train_data_culculation(self, time_train_data):
        fft_result = torch.fft.rfft(time_train_data)
        real = torch.real(fft_result)[1:]  # k=0时模型频域恒等于0，不需要算
        imag = torch.imag(fft_result)[1:]
        result = torch.stack((real, imag), dim=1).type(torch.float32)
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
        
        
        N = self.sequence_len
        kc = kc.repeat(1,len(self.trainable_k))
        k = self.trainable_k.repeat(kc.shape[0],1)
        phi = self.trainable_phases.repeat(kc.shape[0],1)
        A = self.trainable_amplitudes.repeat(kc.shape[0],1)
        
        # 当两个频率距离很近时，把较低的振幅关闭（乘0）。【问】用损失函数不行吗？【答】不够，因为基础损失函数依旧会一直把振幅往上拉
        # k_detach = self.trainable_k.detach()
        # distance = torch.abs(k_detach.unsqueeze(1) - k_detach.unsqueeze(0)) 
        # index = ( distance < 0.1 ).fill_diagonal_(False) # 对角线不算，因为这是跟自己的距离
        # 【未完成】计算复杂度太高了
        # print(kc)
        ddddd = torch.abs(k.detach()-kc)
        index = (ddddd < 10)  # 为了优化计算速度，对于kc，只取其左右±10的k值参与叠加，因为再远一些频谱泄漏已经很小了

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

        index = index.sum(dim=1)
        index = torch.cumsum(index.cpu(),dim=0)
        real_imag = torch.tensor_split(real_imag,index[:-1])
        real_imag = torch.nn.utils.rnn.pad_sequence(real_imag, batch_first=True)

        real_imag = real_imag.sum(dim=1)
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
            A = self.trainable_amplitudes.abs()
            A_data = A.detach().unfold(0,scale,1)
            A_param = A.unfold(0,scale,1)
            mean = torch.mean(A_data,dim=1,keepdim=True)
            std = torch.std(A_data,dim=1,keepdim=True)
            threshold = mean+std
            loss = torch.sum(A_param[A_data<=threshold]**2)/scale
            # print("frequency_aggregation_loss_function: ", loss.item())
            return loss
        
        def keep_k_stay():  # 可训练参数k会到处乱跑，这里把它限制在初始位置的±2以内
            distance = self.trainable_k - torch.arange(1,self.sequence_len//2+1,dtype=torch.float32,device=self.device)
            loss = (relu(distance-2)+relu(-distance-2))**2 * 10
            loss = torch.sum(loss)
            # k不得小于1
            loss += (relu(1.4-self.trainable_k[:10])**4).sum()
            # print("keep_k_stay: ", loss.item())
            return loss

        def keep_k_separated():
            # 当两个k值靠太近时，把较低的振幅挪给较高的振幅
            k = self.trainable_k
            k1 = k.detach().unsqueeze(1)
            k2 = k.detach().unsqueeze(0)
            A = self.trainable_amplitudes
            A1 = A.unsqueeze(1)
            A2 = A.unsqueeze(0)
            k_distance = torch.abs(k1-k2)  # k_distance没有梯度
            A_distance = torch.abs(A1-A2)
            index = (k_distance<0.8).fill_diagonal_(False) # 筛选掉距离明显大的，减少计算量。对角线不算，因为这是跟自己的距离
            # little = A.detach().unsqueeze(0).repeat(A.shape[0],1) < A.detach().mean()+A.detach().abs() # 振幅本来就很小的，不需要相互补偿
            # A_index = index &(~(little & little.T))  # 减数或被减数很小，都不需要补偿
            A_index = index
            A_distance = A_distance[A_index] 
            weight = 1/(50* k_distance[A_index]**2 + 0.2)
            A_loss = -torch.sum(A_distance*weight) # 把较低的振幅挪给较高的振幅
            
            # 构造损失函数使k值相互远离，但应该注意，对于两个k值，振幅较大的那个原地不动，振幅较小的那个才移动，为此必须选择性关闭梯度
            A_compare = A1.detach() > A2.detach()
            index = index & A_compare # (距离<0.8 且 A1>A2)的 距离索引 
            k_distance = torch.repeat_interleave(k.detach(), index.sum(dim=1)) - k.repeat(k.shape[0],1)[index]
            k_loss = 1/(10* k_distance**2 + 0.05) 
            k_loss = torch.sum(k_loss)*64
            loss = A_loss + k_loss
            # print("keep_k_separated: ", loss.item())
            return loss

        def amplitudes_to_less():  # 让接近0的值直接跑向0
            A = self.trainable_amplitudes.abs()
            A_detach = A.detach()
            # 由于不同的数据振幅分布不同，所以需要先调整数据分布
            # 我们期待的分布是：偏小值位于0~2区间，偏大值趋近正无穷
            threshold = torch.mean(A_detach) # 幅值的分布是偏态分布，有大量的小值，不能直接使用方差归一化
            loss = F.sigmoid((A/threshold)**2)*4 - 2 # 乘4是为了让sigmoid在x=0处导数为1，-2是为了x=0处y=0
            loss = torch.sum(loss)*threshold
            # A/mean是为了调整数据分布，*mean是为了平衡梯度值
            # 【失误记录】我一开始使用的损失函数是F.sigmoid((A/mean)**2)*4没有*mean，自以为把数据分布拉到了合适的范围，却不知道梯度也会发生变化
            # print("amplitudes_to_less: ", loss.item())
            return loss

        def total_loss_func(y_pred, y_true):
            loss = basic_loss_func(y_pred, y_true)
            # print("basic_loss_func: ", loss.item())
            loss += 16 * frequency_aggregation_loss_function(9)  # 旁瓣压抑
            loss += 4*amplitudes_to_less()
            loss += keep_k_stay()
            loss += 4*keep_k_separated()
            return loss

        return total_loss_func

    def train(self, epochs=1000, batch_size=256,lr=0.001,save_model_path=None):
        
        frequency_train_data = self.frequency_train_data_culculation(self.train_data)
        kc = torch.arange(1,self.sequence_len//2 +1)
        train_dataset = TensorDataset(kc.unsqueeze(1), frequency_train_data)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = self.regularization_loss_func(torch.nn.MSELoss())
        self.draw('_init',None,frequency_train_data)
        time1 = 0
        time2 = 0
        time3 = 0
        begin_time = time.time()
        for epoch in range(epochs + 1):
            super().train()
            # 在频域上训练
            for kc, frequency_labels in dataloader:
                kc = kc.to(self.device)
                frequency_labels = frequency_labels.to(self.device)
                optimizer.zero_grad()
                
                # tttt = time.time()
                frequency_pred = self.model_output_FREQUENCY(kc)
                # time1 += time.time() - tttt
                
                loss = loss_func(frequency_pred, frequency_labels)
                # tttt = time.time()
                loss.backward()
                # time2 += time.time() - tttt
                # print(loss.item())
                # tttt = time.time()
                optimizer.step()
                # time3 += time.time() - tttt
                # print("time1:", time1, "time2:", time2, "time3:", time3)
            # print("epoch:", epoch, "loss:", loss.item())
            if epoch % 100 == 0:  # 保持振幅为正数，相位在[0,2π]范围内
                with torch.no_grad():
                    self.trainable_phases[self.trainable_amplitudes < 0] += pi
                    self.trainable_phases.remainder_( 2 * pi) # 取余
                    torch.abs_(self.trainable_amplitudes)
            # 在时域上训练
            # train_t = torch.arange(0,train_data.shape[0],dtype=torch.float32,device=self.device).unsqueeze(1)
            # time_pred = self.model_output_TIME(train_t,window_size).squeeze(1)
            # loss = loss_func(time_pred,train_data)
            # loss.backward()
            # optimizer.step()



            if epoch % PRINT_EPOCHS == 0:
                print("epoch:", epoch, "loss:", loss.item())
                if save_model_path is not None:
                    torch.save(self.state_dict(), save_model_path)
                else:
                    torch.save(self.state_dict(), os.path.join(MODEL_DIR, TIME_ID + ".pth"))
                self.draw(epoch,None,frequency_train_data)
        print("耗时:", time.time() - begin_time)
        
    def draw(self
            ,epoch
            ,time_train_data=None
            ,frequency_train_data=None
            ,TIME_FIELD = True  # 时域图
            ,FREQUENCY_FIELD_REAL = 1  # 频域图(实部)
            ,FREQUENCY_FIELD_IMAG = 1  # 频域图(虚部)
            ,MODEL_AMPLITUDES = True  # 模型振幅
            ,MODEL_PHASES = 1  # 模型相位
            ,FFT_RESULT = 1  # 快速傅里叶变换得到的频谱图
            ,PRED_RESULT = 1  # 预测结果
            ):
        '''
        time_train_data和frequency_train_data只接收一条序列，比如请输入self.train_data[0]
        epoch = None 表示直接绘制图像并plt.show()，而不是保存为文件
        幅频图和相频图的横坐标为频率f，而非k
        '''
        
        time_train_data = self.train_data
        if(frequency_train_data is None):
            frequency_train_data = self.frequency_train_data_culculation(time_train_data)
        

        imgs_num = (
            TIME_FIELD
            + FREQUENCY_FIELD_REAL
            + FREQUENCY_FIELD_IMAG
            + MODEL_AMPLITUDES
            + MODEL_PHASES
        )
        imgs_count = 1
        
        
        plt.rcParams['font.sans-serif'] = ['SimHei']   # 中文配置
        plt.rcParams['axes.unicode_minus'] = False  # 负号显示
        
        with torch.no_grad():
            plt.figure(figsize=(25, 5 * imgs_num))
            N = self.sequence_len
            kc = torch.arange(1,N//2 +1)
            frequency_pred = self.model_output_FREQUENCY(kc.to(self.device)).cpu().numpy()
            k_pred = self.trainable_k.detach().cpu().numpy()
            k_label = np.arange(1,N//2 +1)
            frequency_train_data = frequency_train_data.cpu().numpy()
            trainable_amplitudes = self.trainable_amplitudes.detach().cpu().numpy()
            trainable_phases = self.trainable_phases.detach().cpu().numpy()
            if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                fft_pred = torch.fft.rfft(time_train_data)
                fft_amplitudes = torch.abs(fft_pred)*2 / N
                fft_amplitudes = fft_amplitudes.cpu().numpy()
                k_fft = np.arange(0, fft_amplitudes.shape[0])
            if TIME_FIELD:  # 时域
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                train_signal = (time_train_data + self.mean).cpu().numpy()
                plt.plot(np.arange(N), train_signal, label="训练数据的时域", color="blue",zorder=2)
                if self.test_data is not None:
                    plt.plot(self.test_data[0],self.test_data[1], label="真实信号的时域", color="green",zorder=1)
                    t = np.arange(min(-N,min(self.test_data[0])), 
                                  max(2*N,max(self.test_data[0])))
                else:
                    t = np.arange(-N, 2*N)
                if PRED_RESULT:  # 预测结果
                    time_pred = self.model_output_TIME(
                                    torch.tensor(t, dtype=torch.float32, device=self.device).unsqueeze(1)
                                ).cpu().numpy()
                    
                    plt.plot(t, time_pred, label="模型预测的时域", alpha=0.5, color="red",zorder=3)
                    
                plt.grid(True)
                plt.legend(loc='upper right')
            if FREQUENCY_FIELD_REAL:  # 频域图(实部)
                real_pred = frequency_pred[:, 0]
                real_data = frequency_train_data[:, 0]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(k_label, real_data, label="训练数据的频域（实部）", color="green")
                plt.plot(k_label, real_pred, label="模型的频域（实部）", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend(loc='upper right')
            if FREQUENCY_FIELD_IMAG:  # 频域图(虚部)
                imag_pred = frequency_pred[:, 1]  # cos的频域
                imag_data = frequency_train_data[:, 1]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(k_label, imag_data, label="训练数据的频域（虚部）", color="green")
                plt.plot(k_label, imag_pred, label="模型的频域（虚部）", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend(loc='upper right')

            if MODEL_AMPLITUDES:  # 模型振幅
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    plt.scatter(k_fft/N, fft_amplitudes, label="离散傅里叶变换所得幅值", color="blue", marker="s", s=2,alpha=0.7)
                    plt.vlines(k_fft/N, ymin=0, ymax=fft_amplitudes,linewidths=0.7, colors="blue", linestyles="solid",alpha=0.7)
                if PRED_RESULT:  # 预测结果
                    plt.scatter(k_pred/N,trainable_amplitudes,label="模型估计幅值",color="red",marker=".", s=10,alpha=0.7)
                    plt.vlines(k_pred/N,ymin=0,ymax=trainable_amplitudes,linewidths=0.7,colors="red",linestyles="--",alpha=0.7)
                if self.test_frequencies_amplitudes_phases is not None:
                    frequencies = self.test_frequencies_amplitudes_phases[0]
                    amplitudes = self.test_frequencies_amplitudes_phases[1]
                    plt.scatter(frequencies,amplitudes,label="真实幅值",color="green",marker="*",s=10,zorder=-2)
                    plt.vlines(frequencies,ymin=0,ymax=amplitudes,linewidths=0.7,colors="green",linestyles="-",alpha=0.4,zorder=-2)
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
                    plt.scatter(k_fft/N, fft_phases, color="blue", marker="s", s=5,alpha=alpha)
                    plt.scatter([], [], label="离散傅里叶变换所得相位", color="blue", marker="s", s=5,alpha=1) # 用于指定图例样式，因为不透明度不一样
                    plt.vlines(k_fft/N, ymin=0, ymax=fft_phases,linewidths=0.7, colors="blue", linestyles="solid",alpha=alpha)
                
                if PRED_RESULT:  # 预测结果
                    alpha = trainable_amplitudes**2  # 振幅越小，不透明度越低
                    alpha = alpha / np.mean(alpha[alpha>np.mean(alpha)]) * 0.8
                    alpha[alpha > 0.8] = 0.8
                    plt.scatter(k_pred/N,trainable_phases,color="red",marker=".", s=15,alpha=alpha)
                    plt.scatter([], [], label="模型估计相位", color="red",marker=".", s=15, alpha=1) # 用于指定图例样式，因为不透明度不一样
                    plt.vlines(k_pred/N,ymin=0,ymax=trainable_phases,linewidths=0.7,colors="red",linestyles="--",alpha=alpha)
                
                if self.test_frequencies_amplitudes_phases is not None:
                    frequencies = self.test_frequencies_amplitudes_phases[0]
                    phases = self.test_frequencies_amplitudes_phases[2]
                    plt.scatter(frequencies,phases,label="真实相位",color="green",marker="*",s=50,zorder=-2)
                plt.grid(True,zorder=-10)
                plt.legend(loc='upper right')
        if epoch is not None:
            plt.savefig(os.path.join(IMAGE_DIR, f"epoch{epoch}.png"),bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

