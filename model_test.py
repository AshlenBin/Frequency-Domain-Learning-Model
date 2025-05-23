import numpy as np
import torch
from FDLM4 import FDLM

def func2sequence(func, begin, end, noisy=0):
    sequence_data = []
    for t in range(begin, end):
        sequence_data.append(func(t))
    sequence_data = np.array(sequence_data)
    if noisy:
        np.random.seed(1000)  # 为了实验可重复，设置随机种子
        sequence_data += np.random.normal(0, noisy, sequence_data.shape,)
    return sequence_data


def test_ideal():
    # 下面是测试用的理想信号
    # test_T =   90 / np.array([1])  # 周期
    # test_amplitudes = np.array([1], dtype=np.float32)  # 振幅
    # test_phases =     np.array([1])  # 相位
    # noisy = 0
    # sequence_len = 500
    # model_path = "models/单正弦波.pth"
    # -------------------------------
    test_T =   1300 / np.array([2, 5, 10, 42, 45, 50,228,230,240,250,567,570])  # 周期
    test_amplitudes = np.array([4, 2,  4,  3,  2,  5,  2,  2,  3,  2,  3,  4], dtype=np.float32)  # 振幅
    test_phases =     np.array([1, 0,  2,  2,  1,1.2,1.2,0.5,3.2,0.3,2.5,  3])  # 相位
    noisy = 2
    sequence_len = 1400
    model_path = "models/多波有噪声.pth"
    # -------------------------------
    # test_T =    400 / np.array([2, 5, 10, 42, 45, 50])  # 周期
    # test_amplitudes = np.array([4, 2,  4,  3,  2,  5], dtype=np.float32)  # 振幅
    # test_phases =     np.array([1, 0,  2,  2,  1,1.2])  # 相位
    # noisy = 0
    # sequence_len = 500
    # model_path = "models/短序列多波无噪声.pth"
    # -------------------------------

    # 大量随机频率--------------------
    # np.random.seed(1000)  # 为了实验可重复，设置随机种子
    # test_T = 1300 / (np.arange(3,600,4)+np.random.randint(-2,2,size=(600-3)//4 +1) )
    # test_amplitudes = np.abs(np.random.normal(4,3,size=len(test_T))) 
    # test_phases = np.random.uniform(0,2*np.pi,size=len(test_T))
    # noisy = 2
    # sequence_len = 1400
    # model_path = "models/大量随机频率（有噪声）.pth"
    # -------------------------------
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
    test_frequencies_amplitudes_phases = [
        1/test_T,
        test_amplitudes,
        test_phases
    ]

    model = FDLM(train_data,test_data,test_frequencies_amplitudes_phases)

    # model.load_model(model_path)
    model.train(epochs=2000,batch_size=512,lr=0.01,save_model_path=model_path)
    model.draw(epoch=None)
    # model.draw(epoch=None,FFT_RESULT=False)
    # model.draw(epoch=None,FFT_RESULT=True,PRED_RESULT=False)

    exit()
test_ideal()

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

    data = export_sd.export("Re160",filter=True)
    
    test_data = data
    # train_data = torch.tensor(test_data, device="cuda")
    # test_data = [np.arange(len(test_data)),test_data]

    model = FDLM(test_data[150])
    model.load_model(r"models/Re160/Re160[150].pth")
    # model.train(epochs=1000, batch_size=512,lr=0.1,save_model_path="models\Re160.pth")
    model.draw(epoch=None,PRED_RESULT=False)
    exit()
    # for i in range(1000):
    #     train_data = test_data[i]
    #     model = FDLM(train_data)
    #     model.train(epochs=1000, batch_size=512,lr=0.01,save_model_path=f"models/Re160/Re160[{i}].pth")
    # model.draw(epoch=None)
    # exit()
# test_sd()