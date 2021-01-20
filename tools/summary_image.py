import os
import numpy as np
import cv2
from functools import reduce

if __name__ == '__main__':
    orig_img_type = 'mobile_photos'
    orig_img_path = os.path.join('D:/Mission', orig_img_type)
    result_img_path = 'C:/Users/moi/Desktop/comp'
    result_img_path_2 = 'C:/Users/moi/Desktop/comp2'
    summary_out_path = './out'
    fix_img_height = 378
    avg_img_width = 260
    padding_middle = 24
    max_img_inline = 10
    summary_bg_color = [255, 255, 255]

    summary_img_inlines = []

    # img names
    img_names = os.listdir(orig_img_path)
    # img_names_png = list(map(lambda name: name.replace('jpg', 'png'), img_names))

    total_num = len(img_names)
    cur_img_idx = 0

    max_width = max_img_inline * avg_img_width

    while True:
        if result_img_path_2 != '':
            img_inline = np.ones([fix_img_height * 3, max_width, 3]) * np.array(summary_bg_color)
        else:
            img_inline = np.ones([fix_img_height * 2, max_width, 3]) * np.array(summary_bg_color)
        cur_width = 0
        for i in range(cur_img_idx, cur_img_idx + max_img_inline):
            if i >= total_num:
                cur_img_idx = i
                break

            orig_img = cv2.imread(os.path.join(orig_img_path, img_names[i]))
            result_img = cv2.imread(os.path.join(result_img_path, img_names[i]).replace('.jpg', '.png'))

            if result_img_path_2 != '':
                result_img_2 = cv2.imread(os.path.join(result_img_path_2, img_names[i]).replace('.jpg', '.png'))

            oh, ow, _ = orig_img.shape
            rh, rw, _ = result_img.shape
            if oh != rh or ow != rw:
                orig_img = cv2.resize(orig_img, (rw, rh))

            if rh != fix_img_height:
                new_h = fix_img_height
                new_w = round(float(fix_img_height) / rh * rw)
                orig_img = cv2.resize(orig_img, (new_w, new_h))
                result_img = cv2.resize(result_img, (new_w, new_h))
                if result_img_path_2 != '':
                    result_img_2 = cv2.resize(result_img_2, (new_w, new_h))
                rh = new_h
                rw = new_w

            if cur_width + rw > max_width:
                break

            if result_img_path_2 != '':
                img_inline[:fix_img_height, cur_width:cur_width + rw, :] = orig_img
                img_inline[fix_img_height:fix_img_height * 2, cur_width:cur_width + rw, :] = result_img
                img_inline[fix_img_height * 2:, cur_width:cur_width + rw, :] = result_img_2
            else:
                img_inline[:fix_img_height, cur_width:cur_width + rw, :] = orig_img
                img_inline[fix_img_height:, cur_width:cur_width + rw, :] = result_img

            cur_width += rw
            cur_img_idx = i

        cur_img_idx += 1
        summary_img_inlines.append(img_inline)
        if cur_img_idx < total_num - 1:
            img_padding_middle = np.ones([padding_middle, max_width, 3]) * np.array(summary_bg_color)
            summary_img_inlines.append(img_padding_middle)
        else:
            break

    summary_img = reduce(lambda x, y: np.concatenate((x, y), axis=0), summary_img_inlines)

    os.makedirs(summary_out_path, exist_ok=True)
    cv2.imwrite(os.path.join(summary_out_path, f'summary_{orig_img_type}.png'), summary_img)
