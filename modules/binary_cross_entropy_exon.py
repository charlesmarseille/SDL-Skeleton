import torch

m = torch.nn.Sigmoid()
#bce_loss = torch.nn.BCELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

class loss_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        pos = (input >= 0).float()
#        binary_cross_entropy_loss = torch.log(1 + (input - 2 * input * pos).exp()) - input * (target - pos)
#        binary_cross_entropy_loss = bce_loss(m(input), target, reduce=None)
        dicefocalloss = self.dice_focal_loss(input, target)
        loss = torch.sum((dicefocalloss * weights).view(-1), dim=0, keepdim=True)
        ctx.save_for_backward(input, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target, = ctx.saved_variables
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        grad = (torch.sigmoid(input) - target) * weights
        return grad * grad_output, None

lf = loss_func.apply