import torch.nn as nn
import torch
import random
from torchvision import transforms as T
import kornia


# MLP class for projector and predictor------------------------------------------------------------------
def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


# SimSiamMLP class
def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )


# exponential moving average 动量更新类，beta为0.99---------------------------------------------------------
class EMA(object):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old: nn.Module, new: nn.Module):
        o_d = old.state_dict()
        n_d = new.state_dict()
        o_k = o_d.keys()
        for i in o_k:
            o_d[i] = o_d[i] * self.beta + (1 - self.beta) * n_d[i]
        old.load_state_dict(o_d)
        return old


# --------------------------------------------------------------------------------------------------------
def get_module_device(module):
    return next(module.parameters()).device


# batch展平
def flatten(t):
    return t.reshape(t.shape[0], -1)


# 设置是否需要求梯度
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# --------------------------------------------------------------------------------------------------------
class Frame(nn.Module):
    def __init__(self, base_net, augments_size):
        super().__init__()
        self.training = True
        self.back_embedding = False
        self.input_size = augments_size
        self.augment = True
        # default SimCLR augmentation
        self.augments = {'default': default_aug(augments_size)}
        self.base_net = base_net
        self.base_weight = None
        self.base_path = ''
        self.frame_weight = None
        self.frame_path = ''
        self.test_path = ''
        self.blocks = dict({})
        self.epoch = 0
        self.device = get_module_device(self.base_net)
        # self.to(self.device)
        self.sample = torch.randn(2, 3, self.input_size[0], self.input_size[1], device=self.device)
        self.base_net(self.sample)

    def embedding_only(self, back_net=False, epoch=0):
        self.augment = False
        self.training = False
        self.back_embedding = True
        if back_net:
            self.base_net.load_state_dict(torch.load(f'{self.base_path + " " + str(epoch)}-net.pt'))
            return self.base_net

    def set_train(self):
        self.augment = True
        self.training = True
        self.back_embedding = False

    def save_base(self, epoch=0):
        torch.save(self.base_net.state_dict(), f'{self.base_path + " " + str(epoch)}-net.pt')

    def save_frame(self, epoch=0):
        torch.save(self.self.state_dict(), f'{self.frame_path + " " + str(epoch)}-net.pt')

    def load_frame(self, epoch=0):
        self.load_state_dict(torch.load(f'{self.frame_path + " " + str(epoch)}-net.pt'))

    def test_path_return(self, epoch=0, data=None):
        return f'{self.test_path + data + str(epoch)}.xlsx'


# --------------------------------------------------------------------------------------------------------
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


random_erasing = nn.Sequential(T.RandomErasing(), T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),

                                                              std=torch.tensor([0.229, 0.224, 0.225])), )
# 以给定的概率对张量图像或一批张量图像应用随机水平翻转。
Flip = nn.Sequential(
    kornia.augmentation.RandomHorizontalFlip()
)

# 创建一个使用高斯滤波器模糊的随即水平翻转张量
GaussFlip = nn.Sequential(
    kornia.augmentation.RandomHorizontalFlip(),
    kornia.filters.GaussianBlur2d((3, 3), (1.5, 1.5))
)


# 水平翻转高斯模糊加裁剪
def default_aug(size):
    d = torch.nn.Sequential(
        # 应用数据增强实用程序（augmentation utils)
        # 随机改变图像的亮度、对比度、饱和度和色调。
        # 如果图像是torch Tensor，预期的具有 [..., 1 or 3, H, W] 形状，其中 ... 表示任意数量的前导维度。
        # 如果 img 是 PIL Image，则不支持模式“1”、“I”、“F”和具有透明度的模式（alpha 通道）。
        RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p=0.3
        ),
        # 以 p 的概率（默认 0.1）随机将图像转换为灰度。
        T.RandomGrayscale(p=0.2),
        # 以给定的概率随机水平翻转给定的图像
        T.RandomHorizontalFlip(),
        # 使用随机选择的高斯模糊来模糊图像。
        RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p=0.2
        ),
        # 裁剪图像的随机部分并将其调整为给定大小。
        T.RandomResizedCrop((size[0], size[1])),
        # 用均值和标准差对张量图像进行归一化。
        # 均值（序列）：每个通道的均值序列。
        # std（序列）：每个通道的标准偏差序列。
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])),
    )
    return d
