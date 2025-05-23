# 频域学习模型（Frequency-Domain Learning Model）

## 介绍
本模型将深度学习算法与傅立叶变换数学原理相结合，旨在解决工程实践中快速傅立叶变换的“频谱泄漏”问题。

## 使用方式
```python
git clone https://github.com/AshlenBin/Frequency-Domain-Learning-Model.git
cd Frequency-Domain-Learning-Model
python model_test.py
```

`FDLM1.py`到`FDLM3.py`是过去三个版本的FDLM，请直接使用`FDLM4.py`

`FDLM4_parallel.py`是我尝试将计算并行优化以同时训练多个序列，但实测速度并没有提升，废弃。

`model_test.py`可直接运行模型。其中包含三个测试：
- `test_ideal()`：测试理想信号下的谱估计效果，其中包含多个测试项目（单正弦波、多波有噪声、短序列多波无噪声、大量随机频率，等等）可自行添加
- `test_mako()`：测试扫频OCT实验数据下的谱估计效果（如果你没有这些数据可直接删掉它）
- `test_sd()`：测试谱域OCT实验数据下的谱估计效果（如果你没有这些数据可直接删掉它）

`model_test.py`运行完毕后，效果图片存放在`images`目录下，模型存放在`models`目录下。


## 简要原理
![image](https://github.com/user-attachments/assets/c55210a6-49f0-41ff-bc59-71aab4557cee)
![image](https://github.com/user-attachments/assets/c483aa6f-26ae-4965-91b0-dfc426227869)
