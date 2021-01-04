import torch
import torch.nn as nn


def comp_loss(img, pred_matte, gt_matte, mask=None):
    pred_comp = pred_matte * img
    gt_comp = gt_matte * img
    if mask is None:
        return torch.abs(gt_comp - pred_comp).mean()
    else:
        return torch.abs(gt_comp * mask - pred_comp * mask).mean()


def alpha_loss(pred_matte, gt_matte, mask=None):
    if mask is None:
        return torch.abs(gt_matte - pred_matte).mean()
    else:
        return torch.abs(gt_matte * mask - pred_matte * mask).mean()


def class_loss(pred_trimap_prob, gt_trimap_3):
    gt_trimap_type = gt_trimap_3.argmax(dim=1)   # [B, C(type=1), H, W]
    criterion = nn.CrossEntropyLoss()
    return criterion(pred_trimap_prob, gt_trimap_type)


def matting_loss(img, pred_trimap_prob, pred_matte, pred_matte_u, gt_trimap_3, gt_matte, mode):
    mask = gt_trimap_3[:, 1:2, ...]
    mask = mask.detach()

    if mode == 't-net':
        return class_loss(pred_trimap_prob, gt_trimap_3)
    if mode == 'm-net':
        return (0.5 * alpha_loss(pred_matte_u, gt_matte, mask) +
                0.5 * comp_loss(img, pred_matte_u, gt_matte, mask))
    if mode == 'f-net':
        return (0.5 * alpha_loss(pred_matte, gt_matte) +
                0.5 * comp_loss(img, pred_matte, gt_matte))

    mask = (pred_trimap_prob.softmax(dim=1).argmax(dim=1) == 1).float().unsqueeze(dim=1)
    mask = mask.detach()
    # end2end
    return (0.5  * alpha_loss(pred_matte, gt_matte) +
            0.5  * comp_loss(img, pred_matte, gt_matte) +
            0.05 * alpha_loss(pred_matte_u, gt_matte, mask) +
            0.05 * comp_loss(img, pred_matte_u, gt_matte, mask) +
            0.1 * class_loss(pred_trimap_prob, gt_trimap_3))
