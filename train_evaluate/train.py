import torch
from torchvision import models

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


# batch减少了
def train(data_loader, epoch, frame, opt, labeled=True, data_flag=0):
    for i in range(epoch):
        print(f'epoch:{i + 1}')
        step = 0
        for data in data_loader:
            if labeled:
                data = data[data_flag].to(device)
            else:  # 移动数据到cuda
                data = data.to(device)
            loss = frame(data)
            # print(loss)
            # 梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空，
            # 不断累加，累加一定次数后，根据累加的梯度更新网络参数，然后清空梯度，
            # 进行下一次循环。
            opt.zero_grad()
            # 计算当前张量的梯度
            loss.backward()
            # 执行单个优化步骤（参数更新）
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f'epoch:{i+1}, step{step}')

        # save your improved network
        if (i + 1) % 20 == 0:
            frame.save_base(i + 1)


def activate(frame, net=resnet, size=None, lr=3e-4):
    if size is None:
        size = [64, 128]
    net.to(device)
    learner = frame(net, size)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    return learner, opt
