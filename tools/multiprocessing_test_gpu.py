import multiprocessing
import os
import time
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import psutil
import pynvml

from matting import Matting

gpu = True
model = Matting(checkpoint_path='../checkpoints_old/best.pt', gpu=gpu)
model.model.share_memory()
# print('load model')
max_mem_used = 0
max_gpu_used = 0
pynvml.nvmlInit()


def multi_matting(img_dir, out_comp_dir, bg_color, name):
    # print(f'start: {name}')
    img_path = os.path.join(img_dir, name)

    matte, img, trimap = model.matting(img_path, return_img_trimap=True, net_img_size=480, max_size=378)
    cut = model.cutout(img, matte)
    comp = model.composite(cut, np.array(bg_color) / 255.)

    name = name.replace('.jpg', '.png')

    # print(f'end: {name}')

    os.makedirs(out_comp_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_comp_dir, name), cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR))

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    mem_used = psutil.virtual_memory().used
    return mem_used, gpu_used


def callback(used):
    global max_mem_used, max_gpu_used
    used = np.array(used)
    mem_used = used[:, 0]
    gpu_used = used[:, 1]
    max_mem_used = np.max(mem_used)
    max_gpu_used = np.max(gpu_used)


def main(process_num=1):
    global max_mem_used, max_gpu_used
    img_dir = 'D:/Mission/photos'
    out_comp_dir = '../data/predictions/comp'
    bg_color = [33, 150, 243]  # BGR

    img_names = os.listdir(img_dir)
    params = [(img_dir, out_comp_dir, bg_color, name) for name in img_names]
    pool = ThreadPool(processes=process_num)
    pool.starmap_async(func=multi_matting, iterable=params, callback=callback)

    pool.close()
    pool.join()

    return max_mem_used, max_gpu_used


if __name__ == '__main__':
    main()
