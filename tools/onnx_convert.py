import os

import cv2
import torch

from networks.matting_net import MattingNet

if __name__ == '__main__':
    torch.set_flush_denormal(True)

    ckpt_name = 'mini.pt'
    ckpt_path = f'../checkpoints/{ckpt_name}'
    out_dir = f'../checkpoints/onnx/'
    onnx_path = f'{out_dir}/{ckpt_name}'

    os.makedirs(out_dir, exist_ok=True)

    net = MattingNet(pretrain=False)
    net.cuda()
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    example_img = torch.rand(1, 3, 480, 480).cuda()

    torch.onnx.export(net, example_img, onnx_path, output_names=['alpha', 'trimap', 'alpha_u'], opset_version=12)

    model = cv2.dnn.readNet(onnx_path)
    img = cv2.imread('./1.jpg')
    blob = cv2.dnn.blobFromImage(img)
    model.setInput(blob)
    alpha = model.forward(['alpha'])
    cv2.imwrite('./test.png', img)
