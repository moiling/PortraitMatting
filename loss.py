import torch
import torch.nn as nn
import torch.nn.functional as F


def comp_loss(img, pred_matte, gt_fg=None, gt_bg=None, mask=None):
    merge = gt_fg * pred_matte + gt_bg * (1 - pred_matte)
    if mask is None:
        return F.l1_loss(merge, img)
    else:
        return F.l1_loss(merge * mask, img * mask, reduction='sum') / (torch.sum(mask) + 1e-8)


def alpha_loss(pred_matte, gt_matte, mask=None):
    if mask is None:
        return F.l1_loss(gt_matte, pred_matte)
    else:
        return F.l1_loss(gt_matte * mask, pred_matte * mask, reduction='sum') / (torch.sum(mask) + 1e-8)


def class_loss(pred_trimap_prob, gt_trimap_3):
    gt_trimap_type = gt_trimap_3.argmax(dim=1)   # [B, C(type=1), H, W]
    return __class_loss_type(pred_trimap_prob, gt_trimap_type)


def __class_loss_type(pred_trimap_prob, gt_trimap_type):
    criterion = nn.CrossEntropyLoss()
    return criterion(pred_trimap_prob, gt_trimap_type)


def matting_loss(img, pred_trimap_prob, pred_matte, pred_matte_u, pred_fg_u, gt_trimap_3, gt_matte, mode, gt_fg=None, gt_bg=None):
    mask = gt_trimap_3[:, 1:2, ...]
    mask = mask.detach()

    if mode == 't-net':
        return class_loss(pred_trimap_prob, gt_trimap_3)
    if mode == 'm-net' or mode == 'f-net':
        return (0.5 * alpha_loss(pred_matte_u, gt_matte, mask) +
                0.5 * comp_loss(img, pred_matte_u, gt_fg, gt_bg, mask) +
                0.5 * alpha_loss(pred_fg_u, gt_fg, mask))

    mask = (pred_trimap_prob.softmax(dim=1).argmax(dim=1) == 1).float().unsqueeze(dim=1)
    mask = mask.detach()
    # end2end
    return (0.5  * alpha_loss(pred_matte, gt_matte) +
            0.5  * comp_loss(img, pred_matte, gt_fg, gt_bg) +
            0.5  * alpha_loss(pred_fg_u, gt_fg, mask) +
            0.05 * alpha_loss(pred_matte_u, gt_matte, mask) +
            0.05 * comp_loss(img, pred_matte_u, gt_fg, gt_bg, mask) +
            0.1 * class_loss(pred_trimap_prob, gt_trimap_3))


def correction_loss(pred_trimap_prob, pred_matte):
    pred_matte = pred_matte.detach()
    pred_trimap_type = pred_trimap_prob.detach().softmax(dim=1).argmax(dim=1)    # b=0 u=1 f=2
    correction_region = (pred_matte != 0 & pred_trimap_type == 0) | (pred_matte != 1 & pred_trimap_type == 2)
    target_trimap_type = pred_trimap_type
    target_trimap_type[correction_region] = 1
    return __class_loss_type(pred_trimap_prob, target_trimap_type)
