import os
import time
import torch
import logging

import torchvision


def save_checkpoint(checkpoint_dir, epoch, losses, model, optimizer, mode, best=False, logger=logging.getLogger('utils')):
    """Save a checkpoint."""

    checkpoint = {
        'epoch': epoch,
        'losses': losses,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if best:
        name = f'{mode}-best-epoch-{epoch}-{int(time.time())}.pt'
    else:
        name = f'{mode}-epoch-{epoch}-{int(time.time())}.pt'

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, name)
    logger.debug(f'Saving checkpoint to "{path}"')
    torch.save(checkpoint, path)


def resume_model(model, optimizer, checkpoint_dir, mode, logger=logging.getLogger('utils')):
    checkpoint = load_checkpoint(checkpoint_dir, mode)
    if checkpoint is None:
        return 0, [1e5]

    model.load_state_dict(checkpoint['model_state_dict'])

    if checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    losses = checkpoint['losses']
    logger.debug(f'Load Checkpoint => losses: {losses}, epoch: {start_epoch - 1}')

    return start_epoch, losses


def load_checkpoint(checkpoint_dir, mode, logger=logging.getLogger('utils')):
    """Fetch and load the best checkpoint if it exists."""
    best_model = None
    all_models, best_models = [], []

    if not os.path.exists(checkpoint_dir):
        return None

    for name in os.listdir(checkpoint_dir):
        # if name.startswith(args.mode):
        if 'best' in name:
            best_models.append(name)
        else:
            all_models.append(name)

    # get last model.
    if best_models:
        best_models.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
        best_model = best_models[-1]
    elif all_models:
        all_models.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
        best_model = all_models[-1]

    if best_model:
        path = os.path.join(checkpoint_dir, best_model)
        logger.debug(f'Loading checkpoint from "{path}"')
        checkpoint = torch.load(path)

        # different mode => initial losses & epoch & optimizer_state_dict.
        if not best_model.startswith(mode):
            checkpoint['losses'] = [1e5]
            checkpoint['epoch'] = 0
            checkpoint['optimizer_state_dict'] = None
        return checkpoint
    
    return None


def save_images(out_dir, names, pred_mattes, pred_trimaps_prob, logger=logging.getLogger('utils')):
    """Save a batch of images."""
    matte_path = os.path.join(out_dir, 'matte')
    trimap_path = os.path.join(out_dir, 'trimap')

    os.makedirs(matte_path, exist_ok=True)
    os.makedirs(trimap_path, exist_ok=True)

    # logger.debug(f'Saving {len(names)} images to {out_dir}')

    for idx, name in enumerate(names):
        matte = pred_mattes[idx]
        save_path = os.path.join(matte_path, name)
        torchvision.utils.save_image(matte, save_path)

        trimap = pred_trimaps_prob[idx]
        trimap = trimap.softmax(dim=0)
        trimap = trimap.argmax(dim=0)
        trimap = trimap / 2.
        save_path = os.path.join(trimap_path, name)
        torchvision.utils.save_image(trimap, save_path)
