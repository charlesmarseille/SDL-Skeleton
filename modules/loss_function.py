import torch
import torch.nn.functional as F
m = torch.nn.Sigmoid()
#bce_loss = torch.nn.BCELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()


class loss_func(torch.autograd.Function):
    def __init__(self, weight=None, size_average=True):
        super(loss_func, self).__init__()
        self.loss = None

    @staticmethod
    def forward(ctx, inputs, targets):
        beta = 1 - torch.mean(targets)
        weights = 1 - beta + (2 * beta - 1) * targets

        #pos = (inputs >= 0).float()
        #binary_cross_entropy_loss = torch.log(1 + (inputs - 2 * inputs * pos).exp()) - inputs * (targets - pos)
        binary_cross_entropy_loss = bce_loss(m(inputs), targets)

        #inputs = torch.sigmoid(inputs)     

        # #flatten label and prediction tensors
        #inputs_flat = inputs.view(-1)
        #targets_flat = targets.view(-1)
        
        # #dice
        smooth = 1
        alpha = 0.5
        gamma = 4/3
        intersection = (inputs * targets).sum()        

        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        dice_loss = 1-dice
        #focal
        #BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        BCE = binary_cross_entropy_loss
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
        #loss = torch.sum((dice_loss * weights).view(-1), dim=0, keepdim=True)
        loss = torch.sum((focal_loss *(1 + dice_loss)* weights).view(-1), dim=0, keepdim=True)
        ctx.save_for_backward(inputs, targets)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        inputs, targets, = ctx.saved_variables
        beta = 1 - torch.mean(targets)
        weights = 1 - beta + (2 * beta - 1) * targets
        grad = (torch.sigmoid(inputs) - targets) * weights
        return grad * grad_output, None


lf = loss_func.apply

