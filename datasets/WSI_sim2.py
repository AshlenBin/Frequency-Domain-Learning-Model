"""  WSI生成
    1、按波长随时间均匀分布
    2、按波数随时间均匀分布"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


def normal_diff(f, min_sep, maximum, scale=0.2 * 1e-3):
    """
    Distance between two frequencies follows a normal distribution
    f: 全inf矩阵，储存光程差
    nf: 光程差个数
    min_sep: 表面物理间隔
    """
    row, col = f.shape
    for j in range(row):
        f[j, 0] = np.random.uniform(0, maximum * 1000) * 1e-3
        for i in range(1, col):
            condition = True
            while condition:
                d = np.random.normal(scale=scale)
                f_new = d + np.sign(d) * min_sep + f[j, i - 1]
                condition = (
                        (f_new <= 0 or f_new >= maximum)
                        or ((np.min(np.abs(f[j] - f_new)) < min_sep)
                            or (np.min(np.abs((f[j] - maximum) - f_new)) < min_sep)
                            or (np.min(np.abs((f[j] + maximum) - f_new)) < min_sep))
                )
            f[j, i] = f_new

    return f


def gen_surf(M, N, num_surf, min_sep, maximum):
    """
    M, N: 小体块长、宽
    num_surf: 表面个数
    min_sep: 分辨率
    maximum: 量程
    """
    # V = np.zeros((M, N, num_surf))
    diff = np.ones((M * N, num_surf)) * np.inf
    diff = normal_diff(diff, min_sep, maximum)
    surf = diff.reshape(num_surf, M, N).reshape(M, N, num_surf)
    # surf = diff.reshape(M, N, num_surf)
    return surf


# 光源：BS-785-1-HP
START = 805 * 1e-9  #                    805 * 1e-9 765 * 1e-9 790 * 1e-9 835 * 1e-9 815 * 1e-9
STOP = 880 * 1e-9  # 光源起始, 终止波长设置 880 * 1e-9 915 * 1e-9 890 * 1e-9 845 * 1e-9 865 * 1e-9
SS = 1 * 1e-9  # 扫频速度, nm/s
SR = STOP - START  # 扫频范围
LambdaC = (STOP + START) / 2  # 中心波长
T = round(SR / SS)  # 扫频周期是75s

# 光源参数
Lambda = START
Det_L = SR
k_min = (2 * np.pi) / (Lambda + Det_L)
k_max = (2 * np.pi) / Lambda
Det_k = k_max - k_min

# 相机：G-419B
PixelSize = 5.5 * 1e-6
M = 200
N = 200
Fov_x = (M - 1) * PixelSize  # 相机水平视场大小
Fov_y = (N - 1) * PixelSize  # 相机处垂直视场大小
x = np.arange(-Fov_x / 2, Fov_x / 2 + PixelSize, PixelSize)
y = np.arange(-Fov_y / 2, Fov_y / 2 + PixelSize, PixelSize)  # 相机尺寸刻度
FrameRate = 20  # 帧率,fps
SN = FrameRate * T  # 拍摄总张数
maximum = SN * (LambdaC ** 2) / (4 * SR)  # 量程
min_sep = (2 * np.log2(2) / np.pi) * (LambdaC ** 2) / SR  # 物理分辨率


# 参考面测试
DL = 30*1e-6
Ref = 0
rows = np.arange(M)
cols = np.arange(N)
X, Y = np.meshgrid(rows, cols)
X = X * PixelSize
Y = Y * PixelSize
# Surf1 = 1e-3 + 0.001 * X + 0.002 * Y
Surf1 = 1e-3
Surf2 = Surf1 + DL
# Surf2 = Surf1 + DL
Surf = np.zeros((M, N, 2))
Surf[:, :, 0] = Surf1
Surf[:, :, 1] = Surf2

## 相机 序列采集
# 按波长均匀分布
# WL = STOP:-SR/(SN-1):START;% 波长
# WN = (2*pi)./WL; % 波数序列
# 按波数均匀分布
# WN = np.linspace((2*np.pi)/START, (2*np.pi)/STOP, int(SN))
WN = np.linspace((2 * np.pi) / STOP, (2 * np.pi) / START, int(SN))

# 表面参数设置
num_surf = 2

Ow = Surf - Ref
# 反射光强设置, 大小为(M/10)*(N/10)*num_surf
I = 1 + 0.01 * np.random.rand(M, N, 2)

# 干涉条纹照片生成，一共length(WN)张
Of_n = np.zeros((M, N, len(WN)))
for idn in range(len(WN)):
    # Of_n[:, :, idn] = np.sqrt(I1) * np.exp(1j*2*WN[idn]*Surf)+np.sqrt(I2) * np.exp(1j*2*WN[idn]*Ref)  # 各表面光矢量叠加;
    # Fringe[:, :, idn] = Of_n[:, :, idn] * np.conj(Of_n[:, :, idn])  # 干涉信号生成，即光矢量与其共轭相乘
    for j in range(num_surf):
        Of_n[:, :, idn] = Of_n[:, :, idn] + I[:, :, j] * np.cos(2 * WN[idn] * Ow[:, :, j])

#　谱域oct的干涉条纹相当于扫频oct数据的一个截面
TimeFringe = Of_n[100, :, :]

# 加噪声
noise = np.random.randn(*TimeFringe.shape)
TimeFringe = TimeFringe + noise

plt.figure()
plt.imshow(TimeFringe)
plt.colorbar()
plt.colormaps()
plt.show()

