import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

'''多分类　focal loss'''
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, average=True):
        """
        focal_loss:, -α(1-yi)**γ *ce_loss(xi,yi)
        """
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.average = average

        if isinstance(alpha, list):
            # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            assert len(alpha) == self.num_classes  
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(self.num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

    # pred: [N, C]  target: [N,]
    def forward(self, pred, target):
        assert pred.dim()==2 and target.dim()==1
        pred = pred.view(-1, pred.size(-1))
        alpha = self.alpha.to(pred.device)
        alpha = alpha.gather(0, target.view(-1))
        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = pred_softmax.clamp(min=0.0001, max=1.0)
        pred_logsoft = torch.log(pred_softmax)
        pred_softmax = pred_softmax.gather(1, target.view(-1, 1))
        pred_logsoft = pred_logsoft.gather(1, target.view(-1, 1))
        loss_cls = -torch.mul(torch.pow((1 - pred_softmax), self.gamma), pred_logsoft)
        loss_cls = torch.mul(alpha, loss_cls.t())
        if self.average:
            loss = loss_cls.mean()
        else:
            loss = loss_cls.sum()
        return loss


'''二分类　focal loss'''
class BiFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, average=True):
        super(BiFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.average = average

    # pred: [N, 1] / [N,]   target: [N, 1] / [N,]
    def forward(self, pred, target):
        pred = nn.Sigmoid()(pred)
        # 展开 pred 和 target, 使pred.size = target.size = (BatchSize,1) 
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1-pred,pred),dim=1)
        
        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：　
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor. 
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)
        
        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
 
        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
