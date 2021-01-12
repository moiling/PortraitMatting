import torch

if __name__ == '__main__':
    old_ckpt = '../checkpoints/end2end-best-epoch-9-1610368207.pt'
    new_ckpt = '../checkpoints/mini.pt'

    ckpt = torch.load(old_ckpt)
    ckpt_new = {'model_state_dict': ckpt['model_state_dict']}
    torch.save(ckpt_new, new_ckpt)
