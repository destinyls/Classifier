import torch
import torch.nn as nn

class LabelSmoothFocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, average=True, smooth_eps=0.01):
        super(LabelSmoothFocalLoss, self). __init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth_eps = smooth_eps
        self.num_classes = num_classes
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.average = average

    def forward(self, logits, label):
        '''
        logits:[b, c, h, w]
        label:[b, h, w]
        '''
        pred = logits.softmax(dim = 1)
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1),-1)   # b,c,h,w => b,c,h*w
            pred = pred.transpose(1,2)                        # b,c,h*w => b,h*w,c
            pred = pred.contiguous().view(-1, pred.size(2))   # b,h*w,c => b*h*w,c
        
        label = label.view(-1, 1)                         # b*h*w,1
        alpha = self.alpha.type_as(pred.data)
        alpha_t = alpha.gather(0, label.view(-1))   # b*h*w
        pt = pred.gather(1, label).view(-1)                  # b*h*w
        diff = (1 - pt) ** self.gamma
        FL = -1 * alpha_t * diff * pt.log()
        if self.smooth_eps > 0:
            lce = -1 * torch.sum(pred.log(), dim=1) / self.num_classes
            loss = (1 - self.smooth_eps) * FL + self.smooth_eps * lce
        if self.average:
            return loss.mean()
        else: 
            return loss.sum()