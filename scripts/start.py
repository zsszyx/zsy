import os
import argparse
import setproctitle

proc_title = "zsy"
setproctitle.setproctitle(proc_title)
parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('--gpu', '-g', type=str, help='gpu运行设备', required=True)
args = vars(parser.parse_args())
gpu = args['gpu']
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
print(f'gpu {gpu} is used')

from my_model.BYOL import BYOL
import train_evaluate.train as train
import train_evaluate.evaluate as eval
import data.market1501 as mk1501


def byol_mk1501():
    learner, opt = train.activate(BYOL, train.resnet, [64, 128])
    dataloader = mk1501.create_market1501(64, mk1501.train_path)
    train.train(data_loader=dataloader, epoch=200, frame=learner, opt=opt)
    qd = mk1501.Market1501(mk1501.query_path)
    gd = mk1501.Market1501(mk1501.gallery_path)
    eval.exam(learner, 200, qd, gd)