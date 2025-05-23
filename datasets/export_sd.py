import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei' # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# 线性插值时用到的一维插值函数
def interpolate_row(row, x, x_new):
    return np.interp(x_new, x, row)


# 非线性矫正：断层扫描数据需要矫正；输入二维矩阵
def interp1(TimeFringe):
    WaveNum = np.size(TimeFringe, axis=1)
    WaveX = np.array(range(WaveNum))
    k_min = 2 * np.pi / (LambdaC + 0.5 * DetLamda) / 10 ** -6  # 设置插值参数
    k_max = 2 * np.pi / (LambdaC - 0.5 * DetLamda) / 10 ** -6
    k = np.linspace(k_min, k_max, len(WaveX))
    x_interp = (2 * np.pi * 10 ** 6 / k - (LambdaC - 0.5 * DetLamda)) * len(WaveX) / DetLamda + 1
    x_interp = x_interp[1:len(x_interp) - 1]
    x = np.array(range(1, len(WaveX) + 1))
    TimeFringe = TimeFringe - np.mean(TimeFringe, axis=1, keepdims=True)  # 去除直流
    NewTimeFringe = TimeFringe
    NewTimeFringe[:, 1:len(x_interp) + 1] = np.apply_along_axis(interpolate_row, axis=1, arr=TimeFringe,
                                                                x=x, x_new=x_interp)  # 按行插值
    return NewTimeFringe



# 参数设置
data_path = r'datasets/experimental_data/SD-OCT'
ref_name = '0'
data_name = 'Re160'
Pix_m = 1292
Pix_n = 964
LambdaC = 820  # 中心波长
DetLamda = 75  # 带宽

def export(data_name,x=None,filter=False):
    # 读图
    # 参考面
    BK1 = np.double(plt.imread(data_path + '/' + ref_name + '.tif'))
    # BK1 = np.tile(BK1, (Pix_n, 1))

    # 光谱图
    BK2 = np.double(plt.imread(data_path + '/' + data_name + '.tif'))
    Coh_Image = BK2[:Pix_n, :Pix_m] #- BK1
    # Coh_Image = Coh_Image / BK1

    # 非线性矫正
    NewTimeFringe = interp1(Coh_Image)
    
    # 滤除前5个高频点
    if filter:
        fft_result = np.fft.rfft(NewTimeFringe)
        fft_result[:,:5] = 0
        NewTimeFringe = np.fft.irfft(fft_result)
    
    return NewTimeFringe


if __name__ == '__main__':
    data = export('Re160',filter=False)
    # 傅里叶变换
    am2 = np.abs(np.fft.rfft(data))
    plt.figure()
    plt.imshow(am2.T, vmin=0, vmax=1000, aspect='auto')

    plt.ylabel("沿波数轴采样点的索引")
    plt.xlabel("沿直线扫描样本位置的索引")
    plt.colorbar()
    plt.show()

    # plt.figure()
    # # plt.imshow(am2.T, vmin=1.5/20, vmax=1.5, aspect='auto')
    # plt.plot(am2[99, :])
    # plt.show()