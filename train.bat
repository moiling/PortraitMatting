@echo off

set img=D:/Dataset/Adobe_Half_Human/comp_bg/image
set tmp=D:/Dataset/Adobe_Half_Human/comp_bg/trimap
set mat=D:/Dataset/Adobe_Half_Human/comp_bg/alpha
set fg=D:/Dataset/Adobe_Half_Human/comp_bg/fg
set bg=D:/Dataset/Adobe_Half_Human/comp_bg/bg
set val_out=data/val
set val_img=D:/Mission/photos
set val_tmp=D:/Mission/removebg_trimap
set val_mat=D:/Mission/removebg_alpha
set ckpt=checkpoints
set patch_size=200
set sample=1000
set epoch=50

python train.py -dgr -m=t-net   --img=%img% --trimap=%tmp% --matte=%mat% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
python train.py -dgr -m=m-net   --img=%img% --trimap=%tmp% --matte=%mat% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
python train.py -dgr -m=f-net   --img=%img% --trimap=%tmp% --matte=%mat% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
python train.py -dgr -m=end2end --img=%img% --trimap=%tmp% --matte=%mat% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%

Pause