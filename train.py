import torch
import random
import argparse

import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from dataloader.dataset import TrainDataset, ValDataset
from dataloader.prefetcher import PreFetcher

"""================================================== Arguments =================================================="""
parser = argparse.ArgumentParser('Portrait Matting Training Arguments.')

parser.add_argument('--img',        type=str,   default='',   help='training images.')
parser.add_argument('--trimap',     type=str,   default='',   help='intermediate trimaps.')
parser.add_argument('--matte',      type=str,   default='',   help='final mattes.')
parser.add_argument('--val-out',    type=str,   default='',   help='val image out.')
parser.add_argument('--val-img',    type=str,   default='',   help='val images.')
parser.add_argument('--val-trimap', type=str,   default='',   help='intermediate val trimaps.')
parser.add_argument('--val-matte',  type=str,   default='',   help='val mattes.')
parser.add_argument('--ckpt-out',   type=str,   default='',   help='checkpoints.')
parser.add_argument('--batch',      type=int,   default=2,    help='input batch size for train')
parser.add_argument('--val-batch',  type=int,   default=1,    help='input batch size for val')
parser.add_argument('--epoch',      type=int,   default=10,   help='number of epochs.')
parser.add_argument('--sample',     type=int,   default=1000, help='number of samples. -1 means all samples.')
parser.add_argument('--lr',         type=float, default=1e-5, help='learning rate while training.')
parser.add_argument('--patch-size', type=int,   default=480,  help='patch size of input images.')
parser.add_argument('--seed',       type=int,   default=42,   help='random seed.')

parser.add_argument('-t', '--random-trimap', action='store_true', help='random generate trimap')
parser.add_argument('-d', '--debug',         action='store_true', help='log for debug.')
parser.add_argument('-g', '--gpu',           action='store_true', help='use gpu.')

parser.add_argument('-m', '--mode', type=str, choices=['end2end', 'm-net', 't-net'], default='end2end', help='working mode.')

args = parser.parse_args()

"""================================================= Presetting ================================================="""
torch.set_flush_denormal(True)  # flush cpu subnormal float.
cudnn.enabled = False
cudnn.benchmark = False  # save GPU memory.
# random seed.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

"""================================================= Training ==================================================="""
"""----- Load Data ---------------------"""
# train
train_data = TrainDataset(args)
train_data_loader = DataLoader(train_data, batch_size=args.batch, drop_last=True, shuffle=True)
train_data_loader = PreFetcher(train_data_loader)
# val
val_data = ValDataset(args)
val_data_loader = DataLoader(val_data, batch_size=args.val_batch)
"""----- Build Model -------------------"""

"""----- Build Optimizer ---------------"""

"""----- Update Learning Rate ----------"""

"""----- Forward G ---------------------"""

"""----- Calculate Loss ----------------"""

"""----- Back Propagate ----------------"""

"""----- Clip Large Gradient -----------"""

"""----- Update Parameters -------------"""

"""----- Write Log and Tensorboard -----"""
