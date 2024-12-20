# import torch as tf
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn
from torchvision.utils import save_image
from utils import one_hot, simplex


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # inputs: [batch_size, 2, height, width]
        # targets: [batch_size, height, width] (0,1 for two classes)

        # If logits is True, use cross_entropy instead of binary_cross_entropy
        if self.logits:
            # Cross entropy takes care of softmax, so we use raw logits
            # Convert targets from [batch_size, height, width] to long type for indexing
            targets = targets.long()

            # Compute the cross entropy loss
            CE_loss = F.cross_entropy(inputs, targets, reduction="none")

            # Take exp of -CE_loss to get pt (probability of correct class)
            pt = torch.exp(-CE_loss)

        else:
            # For probability inputs (already softmax-ed)
            # targets needs to be one-hot encoded for binary_cross_entropy
            targets = (
                F.one_hot(targets, num_classes=inputs.size(1))
                .permute(0, 3, 1, 2)
                .float()
            )
            CE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-CE_loss)

        # Focal loss calculation
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduce:
            return F_loss.mean()
        else:
            return F_loss


# class FocalLoss(nn.Module):
#     #target=[batch,channel,height,width]
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce

#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(
#                 inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class DiceLoss(nn.Module):
    # target=[batch,height,width]
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1

    def forward(self, predict, target):
        if predict.dim() == 4:
            # print("predict size: ", predict.size())
            # 第二维留第二个通道
            if predict.size()[1] == 2:
                predict = predict[:, 1, :, :]

                # 删去第二维
                predict = predict.squeeze(1)

        assert (
            predict.size() == target.size()
        ), f"the size of predict and target must be equal, but got predict size is {predict.size()}, target size is {target.size()}"
        num = predict.size(0)

        # pre = torch.sigmoid(predict).view(num, -1)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (
            (pre * tar).sum(-1).sum()
        )  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class SurfaceLoss:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        # print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        # save
        pc1 = pc[0][0]
        dc1 = dc[0][0]
        img = torch.cat((pc1, dc1), dim=0)
        save_image(
            img,
            "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train13/saveimg2/1.png",
        )
        # print('pc shape',pc.shape)
        # print('dc shape',dc.shape)

        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss


def get_IOU(SR, GT, threshold=0.5):
    SR = threshold < SR
    GT = torch.max(GT) == GT
    TP = (SR == 1) & (GT == 1)
    FP = (SR == 1) & (GT == 0)
    FN = (SR == 0) & (GT == 1)
    IOU = float(torch.sum(TP)) / (float(torch.sum(TP + FP + FN)) + 1e-6)

    return IOU


def get_accuracy(SR, GT, threshold=0.5):
    SR = threshold < SR
    GT = torch.max(GT) == GT
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1)
    acc = float(corr) / float(tensor_size)

    return acc


def get_precision(SR, GT, threshold=0.5):
    SR = threshold < SR
    GT = torch.max(GT) == GT
    TP = (SR == 1) & (GT == 1)
    FP = (SR == 1) & (GT == 0)
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    SE = get_recall(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)
    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_recall(SR, GT, threshold=0.5):
    SR = threshold < SR
    GT = torch.max(GT) == GT
    TP = (SR == 1) & (GT == 1)
    FN = (SR == 0) & (GT == 1)
    RC = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return RC
