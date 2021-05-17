#!/usr/bin/env bash
img=/workspace/Dataset/zt4k/train/image
tmp=/workspace/Dataset/zt4k/train/trimap
mat=/workspace/Dataset/zt4k/train/alpha
fg=/workspace/Dataset/zt4k/train/fg
bg=/workspace/Dataset/zt4k/train/bg
val_out=data/val
val_img=/workspace/Dataset/zt4k/test/image
val_tmp=/workspace/Dataset/zt4k/test/trimap
val_mat=/workspace/Dataset/zt4k/test/alpha
val_fg=/workspace/Dataset/zt4k/test/fg
val_bg=/workspace/Dataset/zt4k/test/bg
ckpt=checkpoints
patch_size=480
sample=1000
t_epoch=20
m_epoch=20
f_epoch=20
e_epoch=100
hr_m_epoch=50
hr_f_epoch=20


python train.py \
      -dgr \
      -m=t-net \
      --img=${img} \
      --trimap=${tmp} \
      --matte=${mat} \
      --fg=${fg} \
      --bg=${bg} \
      --val-out=${val_out} \
      --val-img=${val_img} \
      --val-trimap=${val_tmp} \
      --val-matte=${val_mat} \
      --val-fg=${val_fg} \
      --val-bg=${val_bg} \
      --ckpt=${ckpt} \
      --patch-size=${patch_size} \
      --sample=${sample} \
      --epoch=${t_epoch}

python train.py \
      -dgr \
      -m=m-net \
      --img=${img} \
      --trimap=${tmp} \
      --matte=${mat} \
      --fg=${fg} \
      --bg=${bg} \
      --val-out=${val_out} \
      --val-img=${val_img} \
      --val-trimap=${val_tmp} \
      --val-matte=${val_mat} \
      --val-fg=${val_fg} \
      --val-bg=${val_bg} \
      --ckpt=${ckpt} \
      --patch-size=${patch_size} \
      --sample=${sample} \
      --epoch=${m_epoch}

python train.py \
      -dgr \
      -m=f-net \
      --img=${img} \
      --trimap=${tmp} \
      --matte=${mat} \
      --fg=${fg} \
      --bg=${bg} \
      --val-out=${val_out} \
      --val-img=${val_img} \
      --val-trimap=${val_tmp} \
      --val-matte=${val_mat} \
      --val-fg=${val_fg} \
      --val-bg=${val_bg} \
      --ckpt=${ckpt} \
      --patch-size=${patch_size} \
      --sample=${sample} \
      --epoch=${f_epoch}

python train.py \
      -dgr \
      -m=end2end \
      --img=${img} \
      --trimap=${tmp} \
      --matte=${mat} \
      --fg=${fg} \
      --bg=${bg} \
      --val-out=${val_out} \
      --val-img=${val_img} \
      --val-trimap=${val_tmp} \
      --val-matte=${val_mat} \
      --val-fg=${val_fg} \
      --val-bg=${val_bg} \
      --ckpt=${ckpt} \
      --patch-size=${patch_size} \
      --sample=${sample} \
      --epoch=${e_epoch}

python train.py \
      -dgr \
      --hr \
      -m=m-net \
      --img=${img} \
      --trimap=${tmp} \
      --matte=${mat} \
      --fg=${fg} \
      --bg=${bg} \
      --val-out=${val_out} \
      --val-img=${val_img} \
      --val-trimap=${val_tmp} \
      --val-matte=${val_mat} \
      --val-fg=${val_fg} \
      --val-bg=${val_bg} \
      --ckpt=${ckpt} \
      --patch-size=${patch_size} \
      --sample=${sample} \
      --epoch=${hr_m_epoch}

python train.py \
      -dgr \
      --hr \
      -m=f-net \
      --img=${img} \
      --trimap=${tmp} \
      --matte=${mat} \
      --fg=${fg} \
      --bg=${bg} \
      --val-out=${val_out} \
      --val-img=${val_img} \
      --val-trimap=${val_tmp} \
      --val-matte=${val_mat} \
      --val-fg=${val_fg} \
      --val-bg=${val_bg} \
      --ckpt=${ckpt} \
      --patch-size=${patch_size} \
      --sample=${sample} \
      --epoch=${hr_f_epoch}