from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# 请将实验的bmp图片以类似这样的路径存放：
#   根目录/datasets/experimental_data/mako6/1.bmp
                                        # /2.bmp
                                        # /3.bmp
def bmps2sequence(bmps_folder, x, y, cut=None):
    # 打开 bmps_folder 文件夹下的所有 bmp 图片，获取像素点(x,y)的值，合并为序列返回
    # cut: 截取序列的起止位置。cut=(10,1509)表示截取第10到第1509张bmp（含头含尾）。默认为None，即截取全部序列。
    # bmp图片尺寸：284 × 213
    # 确保坐标在图片尺寸范围内
    if not (0 <= x < 284 and 0 <= y < 213):
        raise ValueError("坐标(x, y)超出图片尺寸范围")
    
    sequence_data = []
    
    # 遍历文件夹中的所有文件
    if(cut is None):
        cut = (1, len(os.listdir(bmps_folder)))
    for i in range(cut[0], cut[1]+1):
        filename = f'{i}.bmp'  # 假设文件名是1.bmp, 2.bmp, 3.bmp等
        img_path = os.path.join(bmps_folder, filename)
        if os.path.exists(img_path):  # 确保文件存在
            with Image.open(img_path) as img:
                # 获取像素点(x, y)的值
                pixel_value = img.getpixel((x, y))
                sequence_data.append(pixel_value)

    sequence_data = np.array(sequence_data, dtype=np.float32)
    return sequence_data


def bmps2sequence_retangle(bmps_folder, x, y, cut=None):
    # 获取一个矩形框里的像素点值，含头不含尾
    if not (0 <= x[0] and x[1] < 284 and 0 <= y[0] and y[1] < 213):
        raise ValueError("坐标(x, y)超出图片尺寸范围")
    
    # 遍历文件夹中的所有文件
    if(cut is None):
        cut = (1, len(os.listdir(bmps_folder)))
    sequence_data = np.zeros((y[1]-y[0],x[1]-x[0],cut[1]+1-cut[0]),dtype=np.float32)
    for i in range(cut[0], cut[1]+1):
        filename = f'{i}.bmp'  # 假设文件名是1.bmp, 2.bmp, 3.bmp等
        img_path = os.path.join(bmps_folder, filename)
        if os.path.exists(img_path):  # 确保文件存在
            
            with Image.open(img_path) as img:
                for _x in range(x[0], x[1]):
                    for _y in range(y[0], y[1]):
                        # 获取像素点(x, y)的值
                        pixel_value = img.getpixel((_x, _y))
                        sequence_data[_y-y[0],_x-x[0],i-cut[0]] = pixel_value

    return sequence_data


def draw_sequence_data(sequence_data):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'SimHei' # 中文显示
    plt.rcParams['axes.unicode_minus'] = False
    # 将序列数据绘制成图像
    plt.plot(sequence_data)
    # sum = np.sum(sequence_data)
    avg = np.mean(sequence_data)
    # plt.plot([0, len(sequence_data)], [sum, sum], label='总和')
    plt.plot([0, len(sequence_data)], [avg, avg], label='平均值')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def export(mako_name,x,y):
    bmps_folder = 'datasets/experimental_data/'+mako_name
    sequence_data = bmps2sequence(bmps_folder, x, y,cut=(10,1509))
    return sequence_data

def export_retangle(mako_name,x,y):
    bmps_folder = 'datasets/experimental_data/'+mako_name
    sequence_data = bmps2sequence_retangle(bmps_folder, x, y,cut=(10,1509))
    return sequence_data

if __name__ == '__main__':
    # 测试代码
    sequence_data = export(mako_name='mako6',x=99,y=0)
    print(sequence_data)
    # draw_sequence_data(sequence_data)
    fft_data = np.fft.rfft(sequence_data)
    A = np.abs(fft_data)
    plt.plot(A)
    plt.show()