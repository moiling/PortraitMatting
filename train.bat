@echo off

set img=D:/Dataset/zt4k/train/image
set tmp=D:/Dataset/zt4k/train/trimap
set mat=D:/Dataset/zt4k/train/alpha
set fg=D:/Dataset/zt4k/train/fg
set bg=D:/Dataset/zt4k/train/bg
set val_out=data/val
set val_img=D:/Dataset/zt4k/test/image
set val_tmp=D:/Dataset/zt4k/test/trimap
set val_mat=D:/Dataset/zt4k/test/alpha
set val_fg=D:/Dataset/zt4k/test/fg
set val_bg=D:/Dataset/zt4k/test/bg
set ckpt=checkpoints
set patch_size=200
set sample=1000
set epoch=10

python train.py -dgr -m=t-net   --img=%img% --trimap=%tmp% --matte=%mat% --fg=%fg% --bg=%bg% --val-fg=%val_fg% --val-bg=%val_bg% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
python train.py -dgr -m=m-net   --img=%img% --trimap=%tmp% --matte=%mat% --fg=%fg% --bg=%bg% --val-fg=%val_fg% --val-bg=%val_bg% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
python train.py -dgr -m=f-net   --img=%img% --trimap=%tmp% --matte=%mat% --fg=%fg% --bg=%bg% --val-fg=%val_fg% --val-bg=%val_bg% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
python train.py -dgr -m=end2end --img=%img% --trimap=%tmp% --matte=%mat% --fg=%fg% --bg=%bg% --val-fg=%val_fg% --val-bg=%val_bg% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
python train.py -dgr -m=m-net   --hr --img=%img% --trimap=%tmp% --matte=%mat% --fg=%fg% --bg=%bg% --val-fg=%val_fg% --val-bg=%val_bg% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%
python train.py -dgr -m=f-net   --hr --img=%img% --trimap=%tmp% --matte=%mat% --fg=%fg% --bg=%bg% --val-fg=%val_fg% --val-bg=%val_bg% --val-out=%val_out% --val-img=%val_img% --val-trimap=%val_tmp% --val-matte=%val_mat% --ckpt=%ckpt% --patch-size=%patch_size% --sample=%sample% --epoch=%epoch%

Pause
