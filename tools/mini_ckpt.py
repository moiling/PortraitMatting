import torch

if __name__ == '__main__':
    old_ckpt = '../checkpoints_old/modnet.pt'
    new_ckpt = '../checkpoints_old/modnet_mini.pt'

    ckpt = torch.load(old_ckpt)
    ckpt_new = {'model_state_dict': ckpt['model_state_dict']}
    torch.save(ckpt_new, new_ckpt)
