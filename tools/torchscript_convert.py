import argparse
import os

import torch

from networks.cutout_modnet import CutoutMODNet
from networks.cutout_net import CutoutNet
from networks.matting_net import MattingNet
from networks.modnet import MODNet

if __name__ == '__main__':
    torch.set_flush_denormal(True)

    parser = argparse.ArgumentParser()
    ckpt_name = 'mini.pt'
    ckpt_path = f'../checkpoints/{ckpt_name}'
    out_dir = f'../checkpoints/jit'
    out_path = f'{out_dir}/{ckpt_name}'

    os.makedirs(out_dir, exist_ok=True)

    net = MattingNet(pretrain=False)
    net.cuda()
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    example_img = torch.rand(1, 3, 480, 480).cuda()
    # example_trimap = torch.rand(1, 3, 480, 480)

    # use trimap version and no trimap version -> 2 torch script.
    # traced_script_module = torch.jit.trace(net, (example_img, example_trimap))
    # output = traced_script_module(torch.ones(2, 3, 480, 480), torch.ones(2, 3, 480, 480))
    traced_script_module = torch.jit.trace(net, example_img)
    output = traced_script_module(torch.ones(1, 3, 480, 480).cuda())

    traced_script_module.save(out_path)
