from tensorboardX import SummaryWriter

import utils
import torch
import random
import logging
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from loss import matting_loss
from dataloader.dataset import TrainDataset, ValDataset
from dataloader.prefetcher import PreFetcher
from networks.matting_net import MattingNet

"""================================================== Arguments ================================================="""
parser = argparse.ArgumentParser('Portrait Matting Training Arguments.')

parser.add_argument('--img',        type=str,   default='',   help='training images.')
parser.add_argument('--trimap',     type=str,   default='',   help='intermediate trimaps.')
parser.add_argument('--matte',      type=str,   default='',   help='final mattes.')
parser.add_argument('--fg',         type=str,   default='',   help='fg for loss.')
parser.add_argument('--bg',         type=str,   default='',   help='bg for loss.')
parser.add_argument('--val-out',    type=str,   default='',   help='val image out.')
parser.add_argument('--val-img',    type=str,   default='',   help='val images.')
parser.add_argument('--val-trimap', type=str,   default='',   help='intermediate val trimaps.')
parser.add_argument('--val-matte',  type=str,   default='',   help='val mattes.')
parser.add_argument('--ckpt',       type=str,   default='',   help='checkpoints.')
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
parser.add_argument('-r', '--resume',        action='store_true', default=True, help='load a previous checkpoint if exists.')

parser.add_argument('-m', '--mode', type=str, choices=['end2end', 'f-net', 'm-net', 't-net'], default='end2end', help='working mode.')

args = parser.parse_args()

"""================================================= Presetting ================================================="""
torch.set_flush_denormal(True)  # flush cpu subnormal float.
cudnn.enabled = True
cudnn.benchmark = True
# random seed.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)-15s] [%(name)s:%(lineno)s] %(message)s')
logger = logging.getLogger('train')
tb_logger = SummaryWriter()
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logger.debug(args)

"""================================================ Load DataSet ================================================"""
# train
train_data = TrainDataset(args)
train_data_loader = DataLoader(train_data, batch_size=args.batch, drop_last=True, shuffle=True)
train_data_loader = PreFetcher(train_data_loader)
# val
val_data = ValDataset(args)
val_data_loader = DataLoader(val_data, batch_size=args.val_batch)

"""================================================ Build Model ================================================="""
model = MattingNet()
if args.gpu and torch.cuda.is_available():
    model.cuda()
else:
    model.cpu()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

start_epoch, losses = 1, []
if args.resume:
    start_epoch, losses = utils.resume_model(model, optimizer, args.ckpt, args.mode, logger)

"""================================================= Main Loop =================================================="""
for epoch in range(start_epoch, args.epoch + 1):
    """--------------- Train ----------------"""
    model.train()
    logger.info(f'Epoch: {epoch}/{args.epoch}')

    for idx, batch in enumerate(train_data_loader):
        """ Load Batch Data """
        img         = batch['img']
        trimap_3    = batch['trimap_3']
        gt_trimap_3 = batch['trimap_3']
        gt_matte    = batch['matte']
        gt_fg       = batch['fg']
        gt_bg       = batch['bg']

        if args.gpu and torch.cuda.is_available():
            img         = img.cuda()
            trimap_3    = trimap_3.cuda()
            gt_trimap_3 = gt_trimap_3.cuda()
            gt_matte    = gt_matte.cuda()
            gt_fg       = gt_fg.cuda()
            gt_bg       = gt_bg.cuda()
        else:
            img         = img.cpu()
            trimap_3    = trimap_3.cpu()
            gt_trimap_3 = gt_trimap_3.cpu()
            gt_matte    = gt_matte.cpu()
            gt_fg       = gt_fg.cpu()
            gt_bg       = gt_bg.cpu()

        if args.mode != 'm-net':
            trimap_3 = None

        """ Forward """
        pm, ptp, pmu = model(img, trimap_3)

        """ Calculate Loss """
        loss = matting_loss(img, ptp, pm, pmu, gt_trimap_3, gt_matte, args.mode, gt_fg, gt_bg)
        """ Back Propagate """
        optimizer.zero_grad()
        loss.backward()
        """ Update Parameters """
        optimizer.step()
        """ Write Log and Tensorboard """
        logger.debug(f'{args.mode}\t Batch: {idx + 1}/{len(train_data_loader.orig_loader)} \t'
                     f'Train Loss: {loss.item():8.5f}')

        step = (epoch - 1) * len(train_data_loader.orig_loader) + idx
        if step % 100 == 0:
            tb_logger.add_scalar('TRAIN/Loss', loss.item(), step)

    """------------ Validation --------------"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_data_loader):
            """ Load Batch Data """
            img         = batch['img']
            trimap_3    = batch['trimap_3']
            gt_trimap_3 = batch['trimap_3']
            gt_matte    = batch['matte']

            if args.gpu and torch.cuda.is_available():
                img         = img.cuda()
                trimap_3    = trimap_3.cuda()
                gt_trimap_3 = gt_trimap_3.cuda()
                gt_matte    = gt_matte.cuda()
            else:
                img         = img.cpu()
                trimap_3    = trimap_3.cpu()
                gt_trimap_3 = gt_trimap_3.cpu()
                gt_matte    = gt_matte.cpu()

            if args.mode != 'm-net':
                trimap_3 = None

            """ Forward """
            pm, ptp, pmu = model(img, trimap_3)

            """ Calculate Loss """
            loss = matting_loss(img, ptp, pm, pmu, gt_trimap_3, gt_matte, args.mode)

            val_loss += loss.item()

            """ Write Log and Save Images """
            logger.debug(f'Batch: {idx + 1}/{len(val_data_loader)} \t' f'Validation Loss: {loss.item():8.5f}')
            utils.save_images(args.val_out, batch['name'], pm, ptp, pmu, logger)

    average_loss = val_loss / len(val_data_loader)
    losses.append(average_loss)

    """ Write Log and Tensorboard """
    tb_logger.add_scalar('TEST/Loss', average_loss, epoch)
    logger.info(f'Loss:{average_loss}')

    """------------ Save Model --------------"""
    if min(losses) == average_loss:
        logger.info('Minimal loss so far.')
        utils.save_checkpoint(args.ckpt, epoch, losses, model, optimizer, args.mode, best=True, logger=logger)
    else:
        utils.save_checkpoint(args.ckpt, epoch, losses, model, optimizer, args.mode, best=False, logger=logger)
