#!/usr/bin/env bash
img=/workspace/Dataset/AiSegment/matting_human_half/pre-12/img
tmp=/workspace/Dataset/AiSegment/matting_human_half/pre/trimap
mat=/workspace/Dataset/AiSegment/matting_human_half/pre-12/FBA/alpha
fg=/workspace/Dataset/Adobe_Half_Human/comp_bg/fg
bg=/workspace/Dataset/Adobe_Half_Human/comp_bg/bg
val_out=data/val
val_img=/workspace/Mission/photos
val_tmp=/workspace/Mission/removebg_trimap
val_mat=/workspace/Mission/removebg_alpha
ckpt=checkpoints
patch_size=480
sample=1000
t_epoch=20
m_epoch=20
f_epoch=20
e_epoch=200

: "
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
      --ckpt=${ckpt} \
      --patch-size=${patch_size} \
      --sample=${sample} \
      --epoch=${f_epoch}
"
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
      --ckpt=${ckpt} \
      --patch-size=${patch_size} \
      --sample=${sample} \
      --epoch=${e_epoch}
