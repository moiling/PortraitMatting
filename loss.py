import torch
import torch.nn as nn


def comp_loss(img, pred_matte, gt_matte):
    pred_comp = pred_matte * img
    gt_comp = gt_matte * img
    return torch.abs(gt_comp - pred_comp).mean()


def alpha_loss(pred_matte, gt_matte):
    return torch.abs(gt_matte - pred_matte).mean()


def class_loss(pred_trimap_prob, gt_trimap_3):
    gt_trimap_type = gt_trimap_3.argmax(dim=1)   # [B, C(type=1), H, W]
    criterion = nn.CrossEntropyLoss()
    return criterion(pred_trimap_prob, gt_trimap_type)


def matting_loss(img, pred_trimap_prob, pred_matte, gt_trimap_3, gt_matte, mode):
    if mode == 't-net':
        return class_loss(pred_trimap_prob, gt_trimap_3)
    if mode == 'm-net':
        return (0.5 * alpha_loss(pred_matte, gt_matte) +
                0.5 * comp_loss(img, pred_matte, gt_matte))
    # end2end
    return (0.5  * alpha_loss(pred_matte, gt_matte) +
            0.5  * comp_loss(img, pred_matte, gt_matte) +
            0.01 * class_loss(pred_trimap_prob, gt_trimap_3))
