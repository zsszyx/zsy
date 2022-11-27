import copy

import torch

import torch.nn.functional as F
from frame import Frame
from frame import EMA
from frame import MLP
from frame import SimSiamMLP

from frame import flatten

from frame import set_requires_grad
from frame import random_erasing


# loss fn， 高维求面loss
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# main class
class BYOL(Frame):
    def __init__(
            self,
            net,
            image_size,
            return_embedding=False,
            training=True,
            augment=True,
            use_smlp=False,
            use_momentum=False
    ):
        super().__init__(net, image_size)
        self.base_net = net
        self.back_embedding = return_embedding
        self.training = training
        self.augment = augment
        self.use_momentum = use_momentum
        self.base_path = 'BYOL_base'
        self.frame_path = 'BYOL_frame'
        self.test_path = 'BYOL_test'
        self.augments['r_erasing'] = random_erasing
        self.blocks['t_base'] = copy.deepcopy(self.base_net)
        self.blocks['o_projector'] = SimSiamMLP if use_smlp else MLP
        self.blocks['t_projector'] = SimSiamMLP if use_smlp else MLP
        self.blocks['predict'] = MLP
        if self.use_momentum:
            self.ema = EMA(0.99)
        else:
            self.ema = EMA(0)
        self.initial(self.sample)
        self.to(self.device)
        # online net

    def initial(self, x):
        _ = self.blocks['t_base'](x)
        x = self.base_net(x)
        x = flatten(x)
        self.blocks['o_projector'] = self.blocks['o_projector'](x.shape[1], 256, 4096)
        self.blocks['t_projector'] = self.blocks['t_projector'](x.shape[1], 256, 4096)
        set_requires_grad(self.blocks['t_projector'], False)
        self.blocks['predict'] = self.blocks['predict'](256, 256, 4096)
        for i in self.blocks.values():
            i.to(self.device)

    # byol 前向传播
    def forward(self, x):
        if not self.training:
            return self.base_net(x)
        assert not (self.training and x.shape[
            0] == 1), '投影层中的batchnorm，训练时必须有大于1个样本'
        # 数据增强
        image_one, image_two = self.augments['default'](x), self.augments['r_erasing'](x)

        online_proj_one, _ = self.blocks['o_projector'](image_one)
        online_proj_two, _ = self.blocks['o_projector'](image_two)

        online_pred_one = self.blocks['predict'](online_proj_one)
        online_pred_two = self.blocks['predict'](online_proj_two)

        # 禁用梯度计算的上下文管理器。
        with torch.no_grad():
            # 更具是否动量更新决定权重迁移方式
            target_encoder = self.ema.update_average(self['t_projector'], self['o_projector'])
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)
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
