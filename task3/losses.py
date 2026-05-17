import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=8):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred, target):
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=8, dice_weight=0.5, ce_weight=0.5, smooth=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth, num_classes=num_classes)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice