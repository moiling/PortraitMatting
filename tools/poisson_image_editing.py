import os
import numpy as np
import cv2

if __name__ == '__main__':
    img_path = 'D:/Mission/photos_png'
    mask_path = '../data/predictions/trimap'
    out_path = './out'
    os.makedirs(out_path, exist_ok=True)

    names = os.listdir(img_path)

    for name in names:
        img_dst = cv2.imread(os.path.join(img_path, name))
        img_src = np.ones_like(img_dst) * np.array([243, 150, 33], dtype=np.uint8)
        img_mask = cv2.imread(os.path.join(mask_path, name))
        img_mask[img_mask > 0] = 255

        center = (int(img_dst.shape[1] / 2), int(img_dst.shape[0] / 2))

        img_clone = cv2.seamlessClone(img_dst, img_src, img_mask, center, cv2.MIXED_CLONE)

        cv2.imwrite(os.path.join(out_path, name), img_clone)
