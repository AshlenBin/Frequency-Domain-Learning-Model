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
PRINT_EPOCHS = 50
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


    

class FDLM(nn.Module):
    def __init__(
        self,
        train_data:torch.Tensor,
        test_data:torch.Tensor=None,
        test_frequencies_amplitudes_phases:torch.Tensor=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        load_model_path=None,
    ):
        """
        train_data: 时域上的训练数据，一行，只输入y值不输入t值，t值默认是0,1,2,...
            例：train_data = [1,2,3,1,2,3,1,2,3,1]
            并行化：train_data = [[1,2,3,1,2,3,1,2,3,1],
                                 [1,2,1,1,2,1,1,2,1,1],
                                 [3,2,3,1,3,2,3,1,3,2]]
        test_data: 时域上的测试数据，两行，第0行t值，第1行y值
            例：test_data = [[-3,-2,-1, 0, 1, 2, 3, 4, 5, 6],
                             [ 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]]
            并行化：test_data = [[[-3,-2,-1, 0, 1, 2, 3, 4, 5, 6],
                                 [ 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]],
                                [[-3,-2,-1, 0, 1, 2, 3, 4, 5, 6],
                                 [ 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]],
                                [[-3,-2,-1, 0, 1, 2, 3, 4, 5, 6],
                                 [ 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]]]
        test_frequencies_amplitudes_phases: 信号的真实频率，幅值，相位
            例：test_frequencies_amplitudes_phases = [
                    1300 / np.array([2, 5, 10, 42, 45, 50,228,230,240,250,567,570]),
                    np.array([4, 2,  4,  3,  2,  5,  2,  2,  3,  2,  3,  4], dtype=np.float64),
                    np.array([1, 0,  2,  2,  1,1.2,1.2,0.5,3.2,0.3,2.5,  3])
                ]
            并行化：test_data = [
                        test_frequencies_amplitudes_phases1,
                        test_frequencies_amplitudes_phases2,
                        test_frequencies_amplitudes_phases3
                    ]
        """
        super().__init__()
        self.device = device
        train_data = torch.as_tensor(train_data, dtype=torch.float64, device=device)
        if(train_data.dim()==1):
            train_data = train_data.unsqueeze(0)
        if(test_data is not None):
            test_data = torch.as_tensor(test_data)
            if(test_data.dim()<2 or test_data.dim()>3):
                raise ValueError("test_data有两行，第0行t值，第1行y值")
            if(test_data.dim()==2):
                test_data = test_data.unsqueeze(0)
            if(test_data.shape[0]!=train_data.shape[0]):
                raise ValueError("train_data和test_data的序列数必须相同")
        self.test_data = test_data
        self.test_frequencies_amplitudes_phases = test_frequencies_amplitudes_phases
        # 对train_data消去常数项
        
        self.mean = torch.mean(train_data,dim=1,keepdim=True)
        self.train_data = train_data - self.mean # 给模型的训练数据在用户输入的数据基础上去掉均值，在输出时再加上
        
        self.sequence_num = train_data.shape[0] # 序列个数
        self.sequence_len = train_data.shape[1] # 每个序列的长度
        self.frequency_num = self.sequence_len//2  # 频率个数,k从1到self.sequence_len//2
        # 为了避免优化器的训练策略让各个序列的可训练参数相互影响，必需分别声明可训练参数，在运算时再用stack把它们拼起来
        self.__trainable_k = nn.ParameterList()  # ParameterList可以让参数自动注册进Module，使得保存和加载模型时能识别到参数
        self.__trainable_amplitudes = nn.ParameterList()
        self.__trainable_phases = nn.ParameterList()
        self.params_dict = []
        fft_preds = torch.fft.rfft(self.train_data)
        fft_amplitudes = torch.abs(fft_preds)*2 / self.sequence_len 
        fft_phases = torch.angle(fft_preds).remainder_( 2 * pi)
        for i in range(self.sequence_num):
            k = nn.Parameter(torch.arange(1,self.sequence_len//2+1,dtype=torch.float64,device=device))  # 可训练参数：频率分量
            amplitudes = nn.Parameter(fft_amplitudes[i][1:].type(torch.float64).to(self.device))  # 可训练参数：振幅
            phases = nn.Parameter(fft_phases[i][1:].type(torch.float64).to(self.device))  # 可训练参数：相位
            self.__trainable_k.append(k)
            self.__trainable_amplitudes.append(amplitudes)
            self.__trainable_phases.append(phases)
            self.params_dict.append({'params': [k, amplitudes, phases]})
        self.stacked_params_renew()
        
        if load_model_path is not None:
            self.load_state_dict(torch.load(load_model_path))
            print("load model from: ", load_model_path)
    
    stacked_trainable_k:torch.Tensor = None
    stacked_trainable_amplitudes:torch.Tensor = None
    stacked_trainable_phases:torch.Tensor = None
    def stacked_params_renew(self):
        # torch.stack在self.__trainable_k发生改变之后并不会自动改变，所以每次迭代之后都要重新计算一次
        self.stacked_trainable_k = torch.stack(list(self.__trainable_k))
        self.stacked_trainable_amplitudes = torch.stack(list(self.__trainable_amplitudes))
        self.stacked_trainable_phases = torch.stack(list(self.__trainable_phases))
    @property
    def trainable_k(self):
        return self.stacked_trainable_k
    @property
    def trainable_amplitudes(self):
        return self.stacked_trainable_amplitudes

    @property
    def trainable_phases(self):
        return self.stacked_trainable_phases


    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print("load model from: ", model_path)
    

    def frequency_train_data_culculation(self, time_train_data):
        fft_result = torch.fft.rfft(time_train_data)
        # fft_result维度是：(序列数,频率点数)
        real = torch.real(fft_result)[:,1:]
        imag = torch.imag(fft_result)[:,1:]
        result = torch.stack((real, imag), dim=2).type(torch.float64)
        result = result.transpose(0,1)  # 转置，变成(频率点数, 序列数, (实部,虚部))
        return result
        # frequency_train_data的维度是：(频率点数, (实部,虚部))
        # 并行化后：(频率点数, 序列数, (实部,虚部))

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
        
        input_len = kc.shape[0]
        N = self.sequence_len
        k = self.trainable_k.unsqueeze(1).repeat(1,input_len,1)  # 维度:(序列个数,kc长度,可训练k)
        phi = self.trainable_phases.unsqueeze(1).repeat(1,input_len,1)
        A = self.trainable_amplitudes.unsqueeze(1).repeat(1,input_len,1) 
        kc = kc.unsqueeze(0).repeat(self.sequence_num,1,self.frequency_num)
        # print(k[0])
        ddddd = torch.abs(k.detach()-kc)
        index = (ddddd < 10)  # 为了优化计算速度，对于kc，只取其左右±10的k值参与叠加，因为再远一些频谱泄漏已经很小了

        k = k[index] 
        kc = kc[index]
        phi = phi[index]
        A = A[index]
        # 【注意】布尔索引会将张量展平，此时k的维度是：
        #   (序列个数×kc长度× ±10的可训练k)，先遍历可训练k，再遍历kc长度，再遍历序列个数
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

        index = index.flatten(0,1).sum(dim=1)
        index = torch.cumsum(index.cpu(),dim=0)
        real_imag = torch.tensor_split(real_imag,index[:-1])
        real_imag = torch.nn.utils.rnn.pad_sequence(real_imag, batch_first=True)
        # real_imag当前维度：(kc长度*序列数,可训练k,(实部,虚部))
        real_imag = real_imag.sum(dim=1)
        # real_imag当前维度：(kc长度*序列数,(实部,虚部))

        real_imag = real_imag.reshape(self.sequence_num,input_len,2).transpose(0,1)
        # real_imag的维度是：(kc的长度, (实部,虚部)) 
        # 并行化后：(kc的长度,序列数,(实部,虚部))
        return real_imag

    def model_output_TIME(self, t: torch.Tensor):
        # t的维度长这样：[[1],[2],[3]]
        t = t * 2 * torch.pi/self.sequence_len
        t = t * self.trainable_k.unsqueeze(1).repeat(1,t.shape[0],1) + self.trainable_phases.unsqueeze(1).repeat(1,t.shape[0],1)
        t = torch.cos(t) @ self.trainable_amplitudes.unsqueeze(2)
        t = t.squeeze(2)
        return t + self.mean

    def regularization_loss_func(self, basic_loss_func=torch.nn.MSELoss()):

        relu = torch.functional.F.relu

        def frequency_aggregation_loss_function(scale):
            # 旁瓣压抑损失函数，在scale区间内，低于μ+σ的振幅值被压低，高于μ+σ的振幅值保持不变
            # scale的值越大，并非使得压抑的范围更大，反而是对偏大值更加宽容，因为扩大范围往往纳入更多偏小值，拉低μ值
            A_detach = self.trainable_amplitudes.detach().unfold(1,scale,1)
            A_train = self.trainable_amplitudes.unfold(1,scale,1)
            mean = torch.mean(A_detach,dim=2,keepdim=True)
            std = torch.std(A_detach,dim=2,keepdim=True)
            threshold = mean+std
            loss = torch.sum(A_train[A_detach<=threshold]**2)/scale
            return loss
        
        def keep_k_stay():  # 可训练参数k会到处乱跑，这里把它限制在初始位置的±2以内
            distance = self.trainable_k - torch.arange(1,self.frequency_num+1,dtype=torch.float64,device=self.device)
            loss = (relu(distance-2)+relu(-distance-2))**2 * 8
            return loss.sum()

        def keep_k_separated():
            # 当两个k值靠太近时，把较低的振幅挪给较高的振幅
            k = self.trainable_k.detach()
            k1 = k.unsqueeze(2)
            k2 = k.unsqueeze(1)
            A1 = self.trainable_amplitudes.unsqueeze(2)
            A2 = self.trainable_amplitudes.unsqueeze(1)
            k_distance = torch.abs(k1-k2)
            A_distance = torch.abs(A1-A2)
            
            index = (k_distance<0.8)# 筛选掉距离明显大的，减少计算量。
            _idx = torch.arange(index.shape[1])
            index[:,_idx,_idx] = False # 对角线不算，因为这是跟自己的距离。维度大于2，fill_diagonal_(False)不能用
            
            A_distance = A_distance[index] 
            weight = 1/(50* k_distance[index]**2 + 0.2)
            A_loss = -torch.sum(A_distance*weight) # 把较低的振幅挪给较高的振幅
            
            # 构造损失函数使k值相互远离，但应该注意，对于两个k值，振幅较大的那个原地不动，振幅较小的那个才移动，为此必须选择性关闭梯度
            A_compare = A1.detach() > A2.detach()
            index = index & A_compare # (距离<0.8 且 A1>A2)的 距离索引 
            k_distance = torch.repeat_interleave(k.flatten(0), index.sum(dim=2).flatten(0)) - self.trainable_k.unsqueeze(1).repeat(1,self.frequency_num,1)[index]
            k_loss = 1/(10* k_distance**2 + 0.05) 
            k_loss = torch.sum(k_loss)*64
            
            return A_loss+k_loss

        def amplitudes_to_less():  # 让接近0的值直接跑向0
            amplitudes = self.trainable_amplitudes.detach()
            # 由于不同的数据振幅分布不同，所以需要先调整数据分布
            # 我们期待的分布是：偏小值位于0~2区间，偏大值趋近正无穷
            mean = torch.mean(amplitudes,dim=1,keepdim=True) # 幅值的分布是偏态分布，有大量的小值，不能直接使用方差归一化
            loss = F.sigmoid((self.trainable_amplitudes/mean)**2)*4 - 2 # 乘4是为了让sigmoid在x=0处导数为1，-2是为了x=0处y=0
            loss = torch.sum(loss)
            return loss

        def total_loss_func(y_pred, y_true):
            loss = basic_loss_func(y_pred, y_true)
            loss += 8 * frequency_aggregation_loss_function(9)  # 旁瓣压抑
            loss += amplitudes_to_less()
            loss += keep_k_stay()
            loss += 10*keep_k_separated()
            return loss

        return total_loss_func



    def train(self, epochs=1000, batch_size=256,lr=0.001,save_model_path=None):
        
        frequency_train_data = self.frequency_train_data_culculation(self.train_data)
        kc = torch.arange(1,self.sequence_len//2 +1)
        train_dataset = TensorDataset(kc.unsqueeze(1), frequency_train_data)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam(self.params_dict, lr=lr)
        loss_func = self.regularization_loss_func(torch.nn.MSELoss())
        self.draw('_init',None,frequency_train_data)
        
        begin_time = time.time()
        time1 = 0
        time2 = 0
        time3 = 0
        for epoch in range(epochs + 1):
            super().train()
            # 在频域上训练
            for kc, frequency_labels in dataloader:
                kc = kc.to(self.device)
                frequency_labels = frequency_labels.to(self.device)
                optimizer.zero_grad()
                self.stacked_params_renew()
                tttt = time.time()
                frequency_pred = self.model_output_FREQUENCY(kc)
                time1 += time.time() - tttt
                
                loss = loss_func(frequency_pred, frequency_labels)
                tttt = time.time()
                loss.backward()
                # print(self.__trainable_amplitudes[0].grad)
                time2 += time.time() - tttt
                # print(loss.item())
                tttt = time.time()
                optimizer.step()
                time3 += time.time() - tttt
                print("time1:", time1, "time2:", time2, "time3:", time3)
            # 在时域上训练
            # train_t = torch.arange(0,self.sequence_len,dtype=torch.float64,device=self.device).unsqueeze(1)
            # time_pred = self.model_output_TIME(train_t,window_size).squeeze(1)
            # loss = loss_func(time_pred,train_data)
            # loss.backward()
            # optimizer.step()

            if epoch % 500 == 0:  # 保持振幅为正数，相位在[0,2π]范围内
                with torch.no_grad():
                    for i in range(self.sequence_num):
                        self.__trainable_phases[i][self.__trainable_amplitudes[i] < 0] += pi
                        self.__trainable_phases[i].remainder_( 2 * pi) # 取余
                        self.__trainable_amplitudes[i].abs_()

            if epoch % PRINT_EPOCHS == 0:
                if save_model_path is not None:
                    torch.save(self.state_dict(), save_model_path)
                else:
                    torch.save(self.state_dict(), os.path.join(MODEL_DIR, TIME_ID + ".pth"))
                print("epoch:", epoch, "loss:", loss.item())
                self.draw(epoch,None,frequency_train_data)
        print("耗时:", time.time() - begin_time)
    def draw(self, epoch, time_train_data=None, frequency_train_data=None,sequence_id=0):
        '''
        time_train_data和frequency_train_data接收全部序列
        sequence_id: 选择绘制哪条序列的频谱图，默认0
        epoch = None 表示直接绘制图像并plt.show()，而不是保存为文件
        幅频图和相频图的横坐标为频率f，而非k
        '''
        
        time_train_data = self.train_data
        if(frequency_train_data is None):
            frequency_train_data = self.frequency_train_data_culculation(time_train_data)
        
        TIME_FIELD = 1  # 时域图
        FREQUENCY_FIELD_REAL = 1  # 频域图(实部)
        FREQUENCY_FIELD_IMAG = 1  # 频域图(虚部)
        MODEL_AMPLITUDES = True  # 模型振幅
        MODEL_PHASES = 1  # 模型相位
        FFT_RESULT = 1  # 快速傅里叶变换得到的频谱图
        PRED_RESULT = 1  # 预测结果

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
            N = self.sequence_len
            kc = torch.arange(1,N//2 +1)
            frequency_pred = self.model_output_FREQUENCY(kc.to(self.device))[:,sequence_id].cpu().numpy()
            k_pred = self.trainable_k[sequence_id].detach().cpu().numpy()
            k_label = np.arange(1,N//2 +1)
            frequency_train_data = frequency_train_data[:,sequence_id].cpu().numpy()
            trainable_amplitudes = self.trainable_amplitudes[sequence_id].detach().cpu().numpy()
            trainable_phases = self.trainable_phases[sequence_id].detach().cpu().numpy()
            if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                fft_pred = torch.fft.rfft(time_train_data[sequence_id])
                fft_amplitudes = torch.abs(fft_pred)*2 / N
                fft_amplitudes = fft_amplitudes.cpu().numpy()
                k_fft = np.arange(0, fft_amplitudes.shape[0])
            if TIME_FIELD:  # 时域
                train_signal = (time_train_data[sequence_id] + self.mean[sequence_id]).cpu().numpy()
                if self.test_data is not None:
                    test_data = self.test_data[sequence_id].cpu().numpy()
                    t = np.arange(min(-N,min(test_data[0])), 
                                  max(2*N,max(test_data[0])))
                else:
                    t = np.arange(-N, 2*N)
                    
                time_pred = self.model_output_TIME(
                                torch.tensor(t, dtype=torch.float64, device=self.device).unsqueeze(1)
                            )[sequence_id].cpu().numpy()
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                # y=0横坐标轴
                if self.test_data is not None:
                    plt.plot(test_data[0],test_data[1], label="true signal", color="green")
                    
                plt.plot(np.arange(N), train_signal, label="train data", color="blue")
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
                imag_pred = frequency_pred[:, 1]  # cos的频域
                imag_data = frequency_train_data[:, 1]
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                plt.plot(k_label, imag_data, label="imag_data", color="green")
                plt.plot(k_label, imag_pred, label="imag_pred", color="red", alpha=0.5)
                plt.grid(True)
                plt.legend(loc='upper right')

            if MODEL_AMPLITUDES:  # 模型振幅
                plt.subplot(imgs_num, 1, imgs_count)
                imgs_count += 1
                if FFT_RESULT:  # 快速傅里叶变换得到的频谱图
                    plt.scatter(k_fft/N, fft_amplitudes, label="fft_amplitudes", color="blue", marker="s", s=2,alpha=0.7)
                    plt.vlines(k_fft/N, ymin=0, ymax=fft_amplitudes,linewidths=0.7, colors="blue", linestyles="solid",alpha=0.7)
                if PRED_RESULT:  # 预测结果
                    plt.scatter(k_pred/N,trainable_amplitudes,label="model_amplitudes",color="red",marker=".", s=10,alpha=0.7)
                    plt.vlines(k_pred/N,ymin=0,ymax=trainable_amplitudes,linewidths=0.7,colors="red",linestyles="--",alpha=0.7)
                if self.test_frequencies_amplitudes_phases is not None:
                    frequencies = self.test_frequencies_amplitudes_phases[sequence_id][0]
                    amplitudes = self.test_frequencies_amplitudes_phases[sequence_id][1]
                    plt.scatter(frequencies,amplitudes,label="true_amplitudes",color="green",marker="*",s=10,zorder=-2)
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
                    plt.scatter([], [], label="fft_phases", color="blue", marker="s", s=5,alpha=1) # 用于指定图例样式，因为不透明度不一样
                    plt.vlines(k_fft/N, ymin=0, ymax=fft_phases,linewidths=0.7, colors="blue", linestyles="solid",alpha=alpha)
                
                if PRED_RESULT:  # 预测结果
                    alpha = trainable_amplitudes**2  # 振幅越小，不透明度越低
                    alpha = alpha / np.mean(alpha[alpha>np.mean(alpha)]) * 0.8
                    alpha[alpha > 0.8] = 0.8
                    plt.scatter(k_pred/N,trainable_phases,color="red",marker=".", s=15,alpha=alpha)
                    plt.scatter([], [], label="model_phases", color="red",marker=".", s=15, alpha=1) # 用于指定图例样式，因为不透明度不一样
                    plt.vlines(k_pred/N,ymin=0,ymax=trainable_phases,linewidths=0.7,colors="red",linestyles="--",alpha=alpha)
                
                if self.test_frequencies_amplitudes_phases is not None:
                    frequencies = self.test_frequencies_amplitudes_phases[sequence_id][0]
                    phases = self.test_frequencies_amplitudes_phases[sequence_id][2]
                    plt.scatter(frequencies,phases,label="true_phases",color="green",marker="*",s=50,zorder=-2)
                plt.grid(True,zorder=-10)
                plt.legend(loc='upper right')
        if epoch is not None:
            plt.savefig(os.path.join(IMAGE_DIR, f"epoch{epoch}.png"),bbox_inches="tight")
            plt.close()
        else:
            plt.tight_layout()
            plt.show()




def test_ideal():
    # 下面是测试用的理想信号
    # test_T =   1300 / np.array([2, 5, 10, 42, 45, 50,228,230,240,250,567,570])  # 周期
    # test_amplitudes = np.array([4, 2,  4,  3,  2,  5,  2,  2,  3,  2,  3,  4], dtype=np.float64)  # 振幅
    # test_phases =     np.array([1, 0,  2,  2,  1,1.2,1.2,0.5,3.2,0.3,2.5,  3])  # 相位
    # noisy = 2
    # sequence_len = 1400
    # -------------------------------
    # test_T =    400 / np.array([2, 5, 10, 42, 45, 50])  # 周期
    # test_amplitudes = np.array([4, 2,  4,  3,  2,  5], dtype=np.float64)  # 振幅
    # test_phases =     np.array([1, 0,  2,  2,  1,1.2])  # 相位
    # noisy = 0
    # sequence_len = 500
    # -------------------------------

    # 大量随机频率--------------------
    np.random.seed(1000)  # 为了实验可重复，设置随机种子
    test_T = 1300 / (np.arange(3,600,4)+np.random.randint(-2,2,size=(600-3)//4 +1) )
    test_amplitudes = np.abs(np.random.normal(4,3,size=len(test_T))) 
    test_phases = np.random.uniform(0,2*np.pi,size=len(test_T))
    noisy = 0
    sequence_len = 1400
    # -------------------------------
    def func2sequence(func, begin, end, noisy=0):
        sequence_data = []
        for t in range(begin, end):
            sequence_data.append(func(t))
        sequence_data = np.array(sequence_data)
        if noisy:
            np.random.seed(1000)  # 为了实验可重复，设置随机种子
            sequence_data += np.random.normal(0, noisy, sequence_data.shape,)
        return sequence_data 
    
    def signal_func(t):
        T = test_T
        amplitudes = test_amplitudes
        phases = test_phases
        return np.sum(amplitudes * np.cos(2 * np.pi * t / T + phases), axis=0)
    
    train_data = func2sequence(signal_func,0,sequence_len,noisy=noisy)
    test_data = np.array([
        np.arange(-sequence_len,sequence_len*2),
        func2sequence(signal_func,-sequence_len,sequence_len*2)
    ])
    test_frequencies_amplitudes_phases = np.array([
        1/test_T,
        test_amplitudes,
        test_phases
    ])

    train_data = np.stack([train_data,train_data],axis=0)
    test_data = np.stack([test_data,test_data],axis=0)
    test_frequencies_amplitudes_phases = np.stack([test_frequencies_amplitudes_phases,test_frequencies_amplitudes_phases],axis=0)

    model = FDLM(train_data,test_data,test_frequencies_amplitudes_phases)
    # model.load_model(r"models\20250424_214140.pth")
    model.train(epochs=4000,batch_size=512,lr=0.01)
    model.draw(epoch=None)

    exit()
# test_ideal()

def test_mako():
    from datasets import export_mako

    data = export_mako.export("mako7", 10, 10)
    test_data = data / np.std(data)
    train_data = torch.tensor(test_data[:1000], device="cuda")
    test_data = [np.arange(len(test_data)),test_data]


    model = FDLM(train_data,test_data)
    model.train(epochs=1000, batch_size=512,lr=0.01)
    # model.load_model(r"models\20250423_214041.pth")
    model.draw(epoch=None)
    exit()
# test_mako()


def test_sd():
    from datasets import export_sd

    data = export_sd.export("Re160")
    test_data = data / np.std(data)
    
    train_data = test_data
    train_data = torch.tensor(test_data[:700], device="cuda")
    test_data = [np.arange(len(test_data)),test_data]
    # for i in range(train_data.shape[0]):
    for i in range(1):
        model = FDLM(train_data[i:i+10])
        model.train(epochs=500, batch_size=512,lr=0.01,save_model_path=f"models/Re160[{i}~{i+10}].pth")
    model.load_model(r"models\Re160.pth")
    model.draw(epoch=None)
    # exit()
test_sd()