import torch
import numpy as np
import torch.utils.data as tua
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


#######################################################################
# Evaluate
def _evaluate(qf: torch.Tensor, ql, qc, gf: torch.Tensor, gl, gc):
    query = qf.view(-1, 1)  # 将查询特征转置
    # print(query.shape)
    ql = np.array(ql)
    qc = np.array(qc)
    gl = np.array(gl)
    gc = np.array(gc)
    score = torch.mm(gf, query)  #
    score = score.squeeze(1).cpu()  # a.squeeze(N) 就是去掉a中指定的维数为一的维度。
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]  # from large to small, 计算相似度得分
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)  # 找出图像标签
    camera_index = np.argwhere(gc == qc)  # 找出相机标签

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)  # 在query中但不在camera中的已排序的唯一值。
    # 同一人但不是同一相机的标签值
    junk_index1 = np.argwhere(gl == -1)  # 找出垃圾标签
    junk_index2 = np.intersect1d(query_index, camera_index)  # 找出同一照相机下的交集
    junk_index = np.append(junk_index2, junk_index1)  # 垃圾标签合并

    ap, CMC_tmp = _compute_mAP(index, good_index, junk_index)

    return ap, CMC_tmp


def _compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()  # 初始化0向量
    if good_index.size == 0:  # 没有好标签
        print('出错')
        cmc[0] = -1
        return ap, cmc

    # 返回一个与index长度相同的布尔数组，该数组为true，其中index的元素不在ar2中，否则为False。
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]  # 选中的垃圾标签被去除了

    # find good_index index
    ngood = len(good_index)  # 好数据长度
    mask = np.in1d(index, good_index)  # 找出标签在index和good——index中的标签

    # [F, F, T, T, F]
    rows_good = np.argwhere(mask)
    # [[2],[3]]
    # 即该函数返回一个折叠成 一维 的数组，返回好标签在选中标签中排在第几个，越靠后精度越差
    rows_good = rows_good.flatten()  # 即该函数返回一个折叠成 一维 的数组
    # [2,3]
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood  # 分批次累计
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)  # 1/1, 2/3, 4/9,分子是第k个正确图像，分母是结果返回了几张
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2  # 计算梯形面积

    return ap, cmc


######################################################################
def _extract(model, qd, gd, path):
    l1 = len(qd)
    l2 = len(gd)
    # 本来应该是64
    # model.to(device)
    query_dataset = tua.DataLoader(dataset=qd, batch_size=64, shuffle=True, drop_last=False)
    gallery_dataset = tua.DataLoader(dataset=gd, batch_size=64, shuffle=True, drop_last=False)
    gallery_feature = []
    gallery_label = np.array([])
    gallery_cam = np.array([])
    for i in gallery_dataset:
        temp_f = model(i[0].to(device))
        gallery_feature.append(temp_f)
        gallery_label = np.append(gallery_label, i[1])
        gallery_cam = np.append(gallery_cam, i[2])
    # t = gallery_feature[-1]
    gallery_feature = torch.cat(gallery_feature, 0)

    m_ap = 0.0
    cmc = torch.IntTensor(len(gallery_label)).zero_()
    # print(len(cmc))
    for i in query_dataset:
        query_feature = model(i[0].to(device))
        query_label = i[1]
        query_cam = i[2]
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for j in range(len(query_label)):
            ap_tmp, CMC_tmp = _evaluate(query_feature[j], query_label[j], query_cam[j], gallery_feature, gallery_label,
                                        gallery_cam)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
        # print(i, CMC_tmp[0])

        m_ap = m_ap + ap
        cmc = cmc + CMC
        #
    cmc = cmc.float()
    cmc = cmc / l1
    m_ap = m_ap / l1
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (cmc[0], cmc[4], cmc[9], m_ap))
    result = pd.DataFrame([float(cmc[0]), float(cmc[1]), float(cmc[2]), float(cmc[4]), float(cmc[9]), m_ap]).transpose()
    result.columns = ["Rank1", "Rank2", "Rank3", "Rank5", "Rank10", "mAP"]

    result.to_excel(path)


def exam(frame, epoch, qd, gd, dataname):
    test_net = frame.embedding_only(back_net=True, epoch=epoch)
    path = frame.test_path_return(epoch, dataname)
    _extract(test_net, qd, gd, path)
