@echo off

set img=D:/Dataset/AiSegment/matting_human_half/pre-12/img
set tmp=D:/Dataset/AiSegment/matting_human_half/pre/trimap
set mat=D:/Dataset/AiSegment/matting_human_half/pre-12/FBA/alpha
set val_out=data/val
set val_img=D:/Mission/photos
set val_tmp=D:/Mission/removebg_trimap
set val_mat=D:/Mission/removebg_alpha
set ckpt=checkpoints
set patch_size=60
set sample=1000

:: python train.py -dgr -m=t-net   --img=%img% --trimap=%tmp% --matte=%mat% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample%
:: python train.py -dgr -m=m-net   --img=%img% --trimap=%tmp% --matte=%mat% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample%
python train.py -dgr -m=end2end --img=%img% --trimap=%tmp% --matte=%mat% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample%

Pause