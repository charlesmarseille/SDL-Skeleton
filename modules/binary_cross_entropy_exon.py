import torch

m = torch.nn.Sigmoid()
#bce_loss = torch.nn.BCELoss()
bce_loss = torch.nn.BCEWithLogitsLoss()

class binary_cross_entropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        pos = (input >= 0).float()
#        binary_cross_entropy_loss = torch.log(1 + (input - 2 * input * pos).exp()) - input * (target - pos)
#        binary_cross_entropy_loss = bce_loss(m(input), target, reduce=None)
        binary_cross_entropy_loss = bce_loss(input, target)
        loss = torch.sum((binary_cross_entropy_loss * weights).view(-1), dim=0, keepdim=True)
        ctx.save_for_backward(input, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target, = ctx.saved_variables
        beta = 1 - torch.mean(target)
        weights = 1 - beta + (2 * beta - 1) * target
        grad = (torch.sigmoid(input) - target) * weights
        return grad * grad_output, None

bce2d = binary_cross_entropy.apply