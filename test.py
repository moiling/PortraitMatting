import utils
import torch
import random
import logging
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader

from loss import matting_loss
from dataloader.dataset import TestDataset
from networks.matting_net import MattingNet

"""================================================== Arguments ================================================="""
parser = argparse.ArgumentParser('Portrait Matting Testing Arguments.')

parser.add_argument('--img',        type=str,   default='',   help='training images.')
parser.add_argument('--trimap',     type=str,   default='',   help='intermediate trimaps.')
parser.add_argument('--matte',      type=str,   default='',   help='final mattes.')
parser.add_argument('--out',        type=str,   default='',   help='val image out.')
parser.add_argument('--ckpt',       type=str,   default='',   help='checkpoints.')
parser.add_argument('--batch',      type=int,   default=1,    help='input batch size for train')
parser.add_argument('--patch-size', type=int,   default=480,  help='patch size of input images.')
parser.add_argument('--seed',       type=int,   default=42,   help='random seed.')

parser.add_argument('-d', '--debug', action='store_true', help='log for debug.')
parser.add_argument('-g', '--gpu',   action='store_true', help='use gpu.')
parser.add_argument('-m', '--mode',  type=str, choices=['end2end', 'f-net', 'm-net', 't-net'], default='end2end', help='working mode.')

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
logger = logging.getLogger('test')
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

logger.debug(args)

"""================================================ Load DataSet ================================================"""
data = TestDataset(args)
data_loader = DataLoader(data, batch_size=args.batch)

"""================================================ Build Model ================================================="""
model = MattingNet()
if args.gpu and torch.cuda.is_available():
    model.cuda()
else:
    model.cpu()

_, _ = utils.resume_model(model, None, args.ckpt, args.mode, logger)

"""------------ Test --------------"""
model.eval()
val_loss = 0
with torch.no_grad():
    for idx, batch in enumerate(data_loader):
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
        logger.debug(f'Batch: {idx + 1}/{len(data_loader)} \t' f'Test Loss: {loss.item():8.5f}')
        utils.save_images(args.out, batch['name'], pm, ptp, pmu, logger)

average_loss = val_loss / len(data_loader)
logger.info(f'average_loss: {average_loss:8.5f}.')
