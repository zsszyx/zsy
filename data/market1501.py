#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import torch.utils.data as tua
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

train_path = 'Market-1501-v15.09.15/bounding_box_train'
query_path = 'Market-1501-v15.09.15/query'
gallery_path = 'Market-1501-v15.09.15/bounding_box_test'


class Market1501(Dataset):
    """
    a wrapper of Market1501 dataset
    """

    def __init__(self, data_path, *args, **kwargs):
        super(Market1501, self).__init__(*args, **kwargs)
        self.data_path = data_path
        # 返回文件列表
        self.imgs = os.listdir(data_path)
        # 选择.jpg文件
        self.imgs = [el for el in self.imgs if os.path.splitext(el)[1] == '.jpg']
        # 身份标签
        self.lb_ids = [int(el.split('_')[0]) for el in self.imgs]
        # 相机标签
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.imgs]
        # 读取所有图片
        self.imgs = [os.path.join(data_path, el) for el in self.imgs]
        # if is_train:
        #     self.trans = transforms.Compose([
        #         transforms.Resize((288, 144)),
        #         transforms.RandomCrop((256, 128)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
        #         RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
        #     ])
        # else:
        # self.trans_tuple = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
        #     ])
        # self.Lambda = transforms.Lambda(lambda crops: [self.trans_tuple(crop) for crop in crops])
        # 改变长宽比，转tensor，根据三个通道均值方差进行标准化
        self.trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),
                                         transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))])

        # useful for sampler
        # 生成标签字典，根据标签返回对应图片索引
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = self.trans(img)
        return img, self.lb_ids[idx], self.lb_cams[idx]

    # ds = Market1501('./Market-1501-v15.09.15/bounding_box_train', is_train = True)
    # im, _, _ = ds[1]
    # print(im.shape)
    # print(im.max())
    # print(im.min())
    # cv2.imshow('erased', im)
    # cv2.waitKey(0)


def create_market1501(batch_size, path):
    dataset = Market1501(path)
    dataloader = tua.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
