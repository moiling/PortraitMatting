import torch
import numpy as np

from PIL import Image
from torchvision.transforms import functional as F
from comp.estimate_fb import estimate_foreground_background
from dataloader import transforms
from networks.inference_matting_net import InferenceMattingNet
from networks.matting_net import MattingNet


class Matting:
    def __init__(self, checkpoint_path='', gpu=False):
        torch.set_flush_denormal(True)  # flush cpu subnormal float.
        self.checkpoint_path = checkpoint_path
        self.gpu = gpu
        self.model = self.__load_model()

    def __load_model(self):
        model = InferenceMattingNet()
        if self.gpu and torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()

        # load checkpoint.
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def matting(self, image_path, return_img_trimap=False, patch_size=480, max_size=-1, trimap_path=None):
        """
        :param   trimap_path:
        :param   image_path:
        :param   max_size:
        :param   return_img_trimap: return origin image and pred_trimap.
        :param   patch_size   : resize to training size for better result. (resize <= 0 => no resize)
        :return:
                 pred_matte : shape: [H, w, 1      ] range: [0, 1]
                 image      : shape: [H, W, RGB(3) ] range: [0, 1]
                 pred_trimap: shape: [H, w, 1      ] range: [0, 1]
        """
        with torch.no_grad():
            image = self.__load_image_tensor(image_path, max_size)
            trimap_3 = self.__load_trimap_tensor(trimap_path, max_size)
            if self.gpu and torch.cuda.is_available():
                image = image.cuda()
                if trimap_3 is not None:
                    trimap_3 = trimap_3.cuda()
            else:
                image = image.cpu()
                if trimap_3 is not None:
                    trimap_3 = trimap_3.cpu()

            b, c, h, w = image.shape

            pred_matte, pred_trimap_prob, pred_matte_u, pred_fg_u = self.model(image, patch_size=patch_size, trimap_3=trimap_3)

            pred_matte = pred_matte.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)
            pred_matte_u = pred_matte_u.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)
            pred_fg_u = pred_fg_u.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)
            image = image.cpu().detach().squeeze(dim=0).numpy().transpose(1, 2, 0)

            if pred_trimap_prob is not None:
                pred_trimap = pred_trimap_prob.squeeze(dim=0).softmax(dim=0).argmax(dim=0)
                pred_trimap = pred_trimap.cpu().detach().unsqueeze(dim=2).numpy() / 2.
            else:
                pred_trimap = trimap_3.argmax(dim=0).cpu().detach().unsqueeze(dim=2).numpy() / 2.

            pred_fg = image.copy()
            pred_fg[np.tile(pred_trimap, 3) < 1] = pred_fg_u[np.tile(pred_trimap, 3) < 1]
            if not return_img_trimap:
                return pred_matte, pred_fg

            return pred_matte, pred_fg, image, pred_trimap, pred_matte_u

    @staticmethod
    def cutout(fg, alpha):
        """
        :param   fg   : shape: [H, W, RGB(3) ] range: [0, 1]
        :param   alpha: shape: [H, w, 1      ] range: [0, 1]
        :return       : shape: [H, W, RGBA(4)] range: [0, 1]
        """
        cutout = np.zeros((fg.shape[0], fg.shape[1], 4))
        cutout[..., :3] = fg[..., ::]
        cutout[...,  3] = alpha.astype(np.float32).squeeze(axis=2)       # [H, W, RGBA(4)]
        return cutout

    @staticmethod
    def composite(cutout, bg):
        """
        :param  cutout: shape: [H, W, RGBA(4)] range: [0, 1]
        :param  bg    : shape: [BGR(3)]        range: [0, 1]
        :return       : shape: [H, W, RGB(3) ] range: [0, 1]
        """
        alpha = cutout[:, :, 3:4]
        fg    = cutout[:, :,  :3]
        image = alpha * fg + (1 - alpha) * bg
        return image

    def __load_image_tensor(self, image_path, max_size=-1):
        image = Image.open(image_path).convert('RGB')
        if max_size > 0:
            [image] = transforms.ResizeIfBiggerThan(max_size)([image])
        [image] = transforms.ToTensor()([image])
        image = image.unsqueeze(dim=0)
        return image

    def __load_trimap_tensor(self, trimap_path, max_size=-1):
        if trimap_path is None:
            return None
        trimap = Image.open(trimap_path).convert('L')

        if max_size > 0:
            [trimap] = transforms.ResizeIfBiggerThan(max_size)([trimap])
        [trimap] = transforms.ToTensor()([trimap])

        # get 3-channels trimap.
        trimap_3 = trimap.repeat(3, 1, 1)
        trimap_3[0, :, :] = (trimap_3[0, :, :] <= 0.1).float()
        trimap_3[1, :, :] = ((trimap_3[1, :, :] < 0.9) & (trimap_3[1, :, :] > 0.1)).float()
        trimap_3[2, :, :] = (trimap_3[2, :, :] >= 0.9).float()

        trimap_3 = trimap_3.unsqueeze(dim=0)
        return trimap_3
