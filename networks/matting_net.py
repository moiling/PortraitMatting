import torch
import torch.nn as nn

from networks.fnet.fusionnet import FusionNet
from networks.mnet.dimnet import DIMNet
from networks.tnet.mobilenet.wrapper import MobileNetWrapper
from networks.tnet.pspnet import PSPNet


class MattingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnet = MobileNetWrapper()
        self.mnet = DIMNet()
        self.fnet = FusionNet()

    def forward(self, img, trimap_3=None):
        """
        :param img:
        :param trimap_3: 3 channels.
        """
        pred_trimap_prob = self.tnet(img)                                 # [B, C(BUF=3),     H, W]
        pred_trimap_softmax = pred_trimap_prob.softmax(dim=1)             # [B, C(BUF=3),     H, W]

        if trimap_3 is None:
            concat = torch.cat([img, pred_trimap_softmax], dim=1)         # [B, C(RGB+BUF=6), H, W]
        else:
            concat = torch.cat([img, trimap_3], dim=1)                    # [B, C(RGB+BUF=6), H, W]

        pred_matte_u = self.mnet(concat)                                  # [B, C(alpha=1),   H, W]
        pred_matte = self.fnet(torch.cat([concat, pred_matte_u], dim=1))  # [B, C(alpha=1),   H, W])

        return pred_matte, pred_trimap_prob, pred_matte_u
