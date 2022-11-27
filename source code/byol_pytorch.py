import copy

from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import random
from torchvision import transforms as T


# helper functions
# 修改默认值
def default(val, def_val):
    return def_val if val is None else val


# batch展平
def flatten(t):
    return t.reshape(t.shape[0], -1)


# 修改对象cache_key的函数值，通过被装饰函数修改
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# 查看模型所处设备
def get_module_device(module):
    return next(module.parameters()).device


# 设置是否需要求梯度
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# loss fn， 高维求面loss
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# augmentation utils 数据增强实用程序， 数据增强模板
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average 动量更新类，beta为0.99
class EMA(object):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# 更新model参数，根据不同的ema_updater参数
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor
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


# 基础神经网络的包装类
# 将管理隐藏层输出的拦截
# 并通过管道将其输入projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2, use_simsiam_mlp=False):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp

        self.hidden = {}
        self.hook_registered = False

    # 查询内部层
    def _find_layer(self):
        if type(self.layer) == str:
            # 返回网络中所有模块的迭代器，产生模块的名称以及模块本身。(str, Module) – 名称和模块的元组
            modules = dict([*self.net.named_modules()])
            # dict.get(key[, value]) key -- 字典中要查找的键。value -- 可选，如果指定键的值不存在时，返回该默认值
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            # 返回直接子模块的迭代器。Iterator[Module]
            children = [*self.net.children()]
            return children[self.layer]
        return None

    # 钩子函数, 记录给定输入所在设备上的展平输出
    def _hook(self, _, inp, output):
        device = inp[0].device
        self.hidden[device] = flatten(output)

    # 将钩子函数注册到指定层前向传播后面
    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        # 在模块上注册一个前向挂钩。 每次forward()计算输出后都会调用该钩子。
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    # 将projectioe修改为指定MLP
    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
        projector.to()
        return projector.to(hidden)

    # 拦截
    def get_representation(self, x):
        # 如果是最后一个隐藏层直接通过网络返回输出
        if self.layer == -1:
            return self.net(x)

        # 注册钩子函数
        if not self.hook_registered:
            self._register_hook()
        # 隐藏层输出字典清楚
        self.hidden.clear()

        # 过神经网络
        _ = self.net(x)
        # 取隐藏层输出
        hidden = self.hidden[x.device]
        # 清除
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} 从未发出过输出'
        return hidden

    # 前向传播
    def forward(self, x, return_projection=True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation
        # 返回z
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


# main class
class BYOL(nn.Module):
    def __init__(
            self,
            net,
            image_size,
            hidden_layer=-2,
            projection_size=256,
            projection_hidden_size=4096,
            augment_fn=None,
            augment_fn2=None,
            moving_average_decay=0.99,
            use_momentum=True,
            return_embedding=False,
            return_projection=True
    ):
        super().__init__()
        self.net = net
        self.return_embedding = return_embedding
        self.return_projection = return_projection

        # default SimCLR augmentation
        # 数据增强
        DEFAULT_AUG = torch.nn.Sequential(
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
            T.RandomResizedCrop((image_size, image_size)),
            # 用均值和标准差对张量图像进行归一化。
            # 均值（序列）：每个通道的均值序列。
            # std（序列）：每个通道的标准偏差序列。
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )
        # 是否更改默认数据增强方式
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)

        # online net
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer,
                                         use_simsiam_mlp=not use_momentum)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # 获取网络设备并包装器放入相同的设备
        device = get_module_device(net)
        self.to(device)

        # 发送模拟图像张量以实例化单例参数
        # [B, C, H, W]
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    # 设置encoder
    @singleton('target_encoder')
    def _get_target_encoder(self):
        # 浅拷贝和深拷贝的区别是：浅拷贝只是将原对象在内存中引用地址拷贝过来了。
        # 让新的对象指向这个地址。而深拷贝是将这个对象的所有内容遍历拷贝过来了，相当于跟原来没关系了，
        # 所以如果你这时候修改原来对象的值跟他没关系了，不会随之更改。
        target_encoder = copy.deepcopy(self.online_encoder)
        # 不求梯度
        set_requires_grad(target_encoder, False)
        return target_encoder

    # 重置target encoder
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    # 更新平均移动
    def update_moving_average(self):
        assert self.use_momentum, '你不需要更新参数迁移方式， 因为您已关闭目标编码器的动量移动'
        assert self.target_encoder is not None, 'target encoder 目标编码器尚未创建'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    # byol 前向传播
    def forward(
            self,
            x,

    ):
        return_embedding = self.return_embedding
        return_projection = self.return_projection
        assert not (self.training and x.shape[
            0] == 1), '由于投影层中的batchnorm，训练时必须有大于1个样本'

        # 直接返回测试输出，不训练
        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        # 数据增强
        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        # 禁用梯度计算的上下文管理器。
        with torch.no_grad():
            # 更具是否动量更新决定权重迁移方式
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            # 其实就相当于变量之间的关系本来是x -> m -> y,这里的叶子tensor是x，但是这个时候对m进行了m.detach_()操作,其实就是进行了两个操作：
            # 将m的grad_fn的值设置为None,这样m就不会再与前一个节点x关联，这里的关系就会变成x, m -> y,此时的m就变成了叶子结点
            # 然后会将m的requires_grad设置为False，这样对y进行backward()时就不会求m的梯度

            target_proj_two.detach_()
            target_proj_one.detach_()
        # 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，
        # 得到的这个tensor永远不需要计算其梯度，不具有grad。
        # 这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播
        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()


random_erasing = nn.Sequential(T.RandomErasing(), T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),

                                                              std=torch.tensor([0.229, 0.224, 0.225])), )
