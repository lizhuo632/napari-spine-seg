import numpy as np
import tifffile as tiff
import torch


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


srpath = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train7/test_data/44_cp_masks.tif"
gtpath = "/share/data/CryoET_Data/lizhuo/SIM-data/actin/train7/test_data/44_mask.tif"
sr = tiff.imread(srpath).astype(np.uint8)
gt = tiff.imread(gtpath).astype(np.uint8)

print("Accuracy:", get_accuracy(torch.tensor(sr), torch.tensor(gt)))
print("Precision:", get_precision(torch.tensor(sr), torch.tensor(gt)))
print("Recall:", get_recall(torch.tensor(sr), torch.tensor(gt)))
print("F1:", get_F1(torch.tensor(sr), torch.tensor(gt)))
print("IOU:", get_IOU(torch.tensor(sr), torch.tensor(gt)))
