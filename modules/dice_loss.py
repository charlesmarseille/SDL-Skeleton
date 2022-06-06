import torch
import torch.nn.functional as F


ALPHA = 0.5
GAMMA = 4/3

class DiceFocalLoss(torch.autograd.Function):
    def __init__(self, weight=None, size_average=True):
        super(DiceFocalLoss, self).__init__()

    @staticmethod
    def forward(ctx, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        ctx.save_for_backward(inputs, targets)
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice = (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  

        
        # BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        # BCE_EXP = torch.exp(-BCE)
        # focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
        

        return 1-dice

    @staticmethod
    def backward(ctx, grad_output):
        input, target, = ctx.saved_variables
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        grad = (torch.sigmoid(input) - target) * weights
        return grad * grad_output, None

dice_focal = DiceFocalLoss.apply