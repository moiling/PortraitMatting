import os
import cv2
import time
import numpy as np
from matting import Matting

if __name__ == '__main__':
    img_dir         = 'D:/Mission/photos'
    out_comp_dir    = 'data/predictions/comp'
    out_matte_dir   = 'data/predictions/matte'
    out_trimap_dir  = 'data/predictions/trimap'
    out_cutout_dir  = 'data/predictions/cutout'
    checkpoint_path = 'checkpoints/end2end-epoch-400-1610256550.pt'

    bg_color = [33, 150, 243]  # BGR

    os.makedirs(out_comp_dir, exist_ok=True)
    os.makedirs(out_trimap_dir, exist_ok=True)
    os.makedirs(out_matte_dir, exist_ok=True)
    os.makedirs(out_cutout_dir, exist_ok=True)

    start_time = time.time()
    M = Matting(checkpoint_path=checkpoint_path, gpu=True)
    end_time = time.time()
    print(f'Load Model Time:{end_time - start_time:.2f}s.')

    for root, ds, fs in os.walk(img_dir):
        for f in fs:
            img_path = os.path.join(root, f)

            start_time = time.time()
            matte, img, trimap = M.matting(img_path, with_img_trimap=True, net_img_size=480, max_size=378)
            end_time = time.time()
            print(f'Matting Time:{end_time - start_time:.2f}s.')

            start_time = time.time()
            cut = M.cutout(img, matte)
            comp = M.composite(cut, np.array(bg_color) / 255.)
            end_time = time.time()
            print(f'Cutout & Composite Time:{end_time - start_time:.2f}s.')

            f = f.replace('.jpg', '.png')

            cv2.imwrite(os.path.join(out_matte_dir, f), np.uint8(matte * 255))
            cv2.imwrite(os.path.join(out_trimap_dir, f), np.uint8(trimap * 255))
            cv2.imwrite(os.path.join(out_comp_dir, f), cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_cutout_dir, f), cv2.cvtColor(np.uint8(cut * 255), cv2.COLOR_RGBA2BGRA))
