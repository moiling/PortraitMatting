@echo off

set img=D:/Mission/photos
set tmp=D:/Mission/removebg_trimap
set mat=D:/Mission/removebg_alpha
set out=data/test
set ckpt=checkpoints

python test.py -dg -m=end2end --img=%img% --trimap=%tmp% --matte=%mat% --out=%out% --ckpt=%ckpt%

Pause
