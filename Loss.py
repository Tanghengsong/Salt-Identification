import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse
import torch
import torch.optim as optim
from functools import partial

class Lovasz_Hinge(nn.Module):
    def __init__(self, per_image=True):
        super(Lovasz_Hinge, self).__init__()
        self.per_image = per_image
        
    def forward(self, logit, truth):
        return lovasz_hinge(logit, truth, per_image=self.per_image)

def mean(l, ignore_nan=False, empty=0):

    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def lovasz_hinge(logits, labels, per_image=True, ignore=None):

    """

    Binary Lovasz hinge loss

      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)

      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)

      per_image: compute the loss per image instead of per batch

      ignore: void class id

    """

    if per_image:

        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))

                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))

    return loss

def lovasz_hinge_flat(logits, labels):

    """
    Binary Lovasz hinge loss

      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)

      labels: [P] Tensor, binary ground truth labels (0 or 1)

      ignore: label to ignore

    """

    if len(labels) == 0:

        # only void pixels, the gradients should be 0

        return logits.sum() * 0.

    signs = 2. * labels.float() - 1.

    errors = (1. - logits * Variable(signs))

    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)

    perm = perm.data

    gt_sorted = labels[perm]

    grad = lovasz_grad(gt_sorted)

    loss = torch.dot(F.relu(errors_sorted), Variable(grad))

    return loss

def flatten_binary_scores(scores, labels, ignore=None):

    """

    Flattens predictions in the batch (binary case)

    Remove labels equal to 'ignore'

    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels

    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]

    return vscores, vlabels

def lovasz_grad(gt_sorted):

    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """

    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union

    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


def mixed_dice_bce_loss(output, target, dice_weight=0.2, dice_loss=None,
                        bce_weight=0.9, bce_loss=None,
                        smooth=0, dice_activation='sigmoid'):

    num_classes = output.size(1)
    target = target[:, :num_classes, :, :].long()
    if bce_loss is None:
        bce_loss = nn.BCEWithLogitsLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss
    return dice_weight * dice_loss(output, target, smooth, dice_activation) + bce_weight * bce_loss(output, target)


def multiclass_dice_loss(output, target, smooth=0, activation='softmax'):
    """Calculate Dice Loss for multiple class output.
    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        activation (string, optional): Name of the activation function, softmax or sigmoid. Defaults to 'softmax'.
    Returns:
        torch.Tensor: Loss value.
    """
    if activation == 'softmax':
        activation_nn = torch.nn.Softmax2d()
    elif activation == 'sigmoid':
        activation_nn = torch.nn.Sigmoid()
    else:
        raise NotImplementedError('only sigmoid and softmax are implemented')

    loss = 0
    dice = DiceLoss(smooth=smooth)
    output = activation_nn(output)
    num_classes = output.size(1)
    target.data = target.data.float()
    for class_nr in range(num_classes):
        loss += dice(output[:, class_nr, :, :], target[:, class_nr, :, :])
    return loss / num_classes

def where(cond, x_1, x_2):
    cond = cond.long()
    return (cond * x_1) + ((1 - cond) * x_2)