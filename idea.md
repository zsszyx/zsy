# BYOL及其衍生
- BYOL -> DINO
    - where a centering
      and sharpening of the momentum teacher outputs is used
      to avoid model collapse.
     
 
- BYOL -> MOBY
    - combines MoCo with BOYL
      to propose a Transformer-specific method called MoBY
****    


# 基础概念
- SSL -> Self-supervised learning


- SUP -> Supervised pre-training


- Fine-tuning
  - 在实践中，由于数据集不够大，很少有人从头开始训练网络。
    常见的做法是使用预训练的网络
    （例如在ImageNet上训练的分类1000类的网络）
    来重新fine-tuning（也叫微调），或者当做特征提取器。
****


# 何凯明idea
- 何凯明-focal loss, 对正负样本sample
****


# 难点
- triplet loss 三元组损失，
  - 它的基本思想是：
    对于设定的三元组(Anchor, Positive, Negative)
    Triplet loss试图学习到一个特征空间，使得在该空间中相同类别的
      基准样本（Anchor）与 正样本（Positive）距离更近，
      不同类别的 Anchor 与负样本（Negative）距离更远。
      - 其借鉴了度量学习中的经典大间隔最近邻
        （Large Margin Nearest Neighbors，LMNN）算法。


- 相互平均教学"（Mutual Mean-Teaching）提供更为可信和稳定的"软"标签