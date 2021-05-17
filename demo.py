import os
import cv2
import time
import numpy as np
from matting import Matting

if __name__ == '__main__':
    img_dir         = 'D:/Mission/photos2'
    trimap_dir      = ''
    out_comp_dir    = 'data/predictions/comp'
    out_matte_dir   = 'data/predictions/matte'
    out_trimap_dir  = 'data/predictions/trimap'
    out_cutout_dir  = 'data/predictions/cutout'
    out_image_dir   = 'data/predictions/image'
    out_matte_u_dir = 'data/predictions/matte_u'
    out_fg_dir      = 'data/predictions/fg'

    checkpoint_path = 'checkpoints/end2end-best-epoch-73-1621231895.pt'

    bg_color = [33, 150, 243]  # RGB

    os.makedirs(out_comp_dir, exist_ok=True)
    os.makedirs(out_trimap_dir, exist_ok=True)
    os.makedirs(out_matte_dir, exist_ok=True)
    os.makedirs(out_cutout_dir, exist_ok=True)
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_matte_u_dir, exist_ok=True)
    os.makedirs(out_fg_dir, exist_ok=True)

    start_time = time.time()
    M = Matting(checkpoint_path=checkpoint_path, gpu=True)
    end_time = time.time()
    print(f'Load Model Time:{end_time - start_time:.2f}s.')

    img_names = os.listdir(img_dir)

    for name in img_names:
        img_path = os.path.join(img_dir, name)
        trimap_path = None
        if trimap_dir is not None and trimap_dir != '':
            trimap_path = os.path.join(trimap_dir, name.replace('.jpg', '.png'))

        start_time = time.time()
        matte, fg, img, trimap, matte_u = M.matting(img_path, return_img_trimap=True, patch_size=480, max_size=1024, trimap_path=trimap_path)
        end_time = time.time()
        print(f'Matting Time:{end_time - start_time:.2f}s.')

        start_time = time.time()
        cut = M.cutout(fg, matte)
        comp = M.composite(cut, np.array(bg_color) / 255.)
        end_time = time.time()
        print(f'Cutout & Composite Time:{end_time - start_time:.2f}s.')

        name = name.replace('.jpg', '.png')

        cv2.imwrite(os.path.join(out_matte_dir, name), np.uint8(matte * 255))
        cv2.imwrite(os.path.join(out_matte_u_dir, name), np.uint8(matte_u * 255))
        cv2.imwrite(os.path.join(out_trimap_dir, name), np.uint8(trimap * 255))
        cv2.imwrite(os.path.join(out_image_dir, name), cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_comp_dir, name), cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_cutout_dir, name), cv2.cvtColor(np.uint8(cut * 255), cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(os.path.join(out_fg_dir, name), cv2.cvtColor(np.uint8(fg * 255), cv2.COLOR_RGBA2BGRA))
