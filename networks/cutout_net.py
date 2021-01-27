import torch
import torch.nn as nn

from comp.estimate_fb_torch import estimate_foreground_background
from networks.fnet.fusionnet import FusionNet
from networks.mnet.dimnet import DIMNet
from networks.tnet.mobilenet.wrapper import MobileNetWrapper
from networks.tnet.pspnet import PSPNet


class CutoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet = PSPNet(pretrain=False)
        self.mnet = DIMNet(pretrain=False)
        self.fnet = FusionNet()

    def forward(self, img, trimap_3=None):
        """
        ONLY BATCH = 1 !!!

        :param img:
        :param trimap_3: 3 channels.
        """
        pred_trimap_prob = self.tnet(img)                                   # [B=1, C(BUF=3),     H, W]
        pred_trimap_softmax = pred_trimap_prob.softmax(dim=1)               # [B=1, C(BUF=3),     H, W]

        if trimap_3 is None:
            concat = torch.cat([img, pred_trimap_softmax], dim=1)           # [B=1, C(RGB+BUF=6), H, W]
        else:
            concat = torch.cat([img, trimap_3], dim=1)                      # [B=1, C(RGB+BUF=6), H, W]

        pred_matte_u = self.mnet(concat)                                    # [B=1, C(alpha=1),   H, W]
        pred_matte = self.fnet(torch.cat([concat, pred_matte_u], dim=1))    # [B=1, C(alpha=1),   H, W]

        fg, _ = estimate_foreground_background(img, pred_matte)             # [B=1, C(RGB=3),     H, W]
        cutout = torch.zeros((1, 4, img.shape[2], img.shape[3]))
        cutout[:, :3, ...] = fg
        cutout[:, 3, ...] = pred_matte                                      # [B=1, C(RGBA=4),    H, W]

        return pred_matte, pred_trimap_prob, cutout, pred_matte_u
