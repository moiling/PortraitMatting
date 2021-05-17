import torch
import torch.nn as nn

from networks.fnet.fusionnet import FusionNet
from networks.mnet.dimnet import DIMNet
from networks.tnet.mobilenet.wrapper import MobileNetWrapper
from torchvision.transforms import functional as F


class InferenceMattingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet = MobileNetWrapper()
        self.mnet = DIMNet()
        self.fnet = FusionNet()

    def forward(self, img, patch_size=-1, trimap_3=None):
        """
        :param img:
        :param patch_size:
        :param trimap_3: 3 channels.
        """
        pred_trimap_prob = None
        if trimap_3 is None:
            resize_img = img
            b, c, h, w = img.shape
            if patch_size > 0:
                resize_img = F.resize(img, [patch_size, patch_size])

            pred_trimap_prob = self.tnet(resize_img)                      # [B, C(BUF=3),     H, W]
            pred_trimap_softmax = pred_trimap_prob.softmax(dim=1)         # [B, C(BUF=3),     H, W]

            if patch_size > 0:
                pred_trimap_prob = F.resize(pred_trimap_prob, [h, w])
                pred_trimap_softmax = F.resize(pred_trimap_softmax, [h, w])

            concat = torch.cat([img, pred_trimap_softmax], dim=1)         # [B, C(RGB+BUF=6), H, W]
        else:
            concat = torch.cat([img, trimap_3], dim=1)                    # [B, C(RGB+BUF=6), H, W]

        pred_u = self.mnet(concat)                                        # [B, C(alpha,FG=4),   H, W]
        pred_matte_u = pred_u[:, 0:1, :, :]
        pred_fg_u = pred_u[:, 1:, :, :]
        pred_matte = self.fnet(torch.cat([concat, pred_matte_u], dim=1))  # [B, C(alpha=1),   H, W]

        return pred_matte, pred_trimap_prob, pred_matte_u, pred_fg_u
