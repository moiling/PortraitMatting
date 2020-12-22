import os
import random
import torch
import dataloader.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
            self,
            img_dir,
            trimap_dir=None,
            matte_dir=None,
            trans=transforms.Compose([transforms.ToTensor()]),
            sample_size=-1,
            random_trimap=False
    ):
        super(BaseDataset, self).__init__()
        self.img_dir = img_dir
        self.trimap_dir = trimap_dir
        self.matte_dir = matte_dir
        self.trans = trans
        self.random_trimap = random_trimap
        self.img_names = []

        for name in os.listdir(self.img_dir):
            self.img_names.append(name)

        if sample_size != -1:
            random.shuffle(self.img_names)
            self.img_names = self.img_names[:sample_size]

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)

        sample = {'name': img_name}

        if (self.trimap_dir is not None or self.random_trimap) and self.matte_dir is not None:
            matte_path = os.path.join(self.matte_dir, img_name).replace('.jpg', '.png')
            matte = Image.open(matte_path).convert('L')

            if self.random_trimap:
                trimap = transforms.GenTrimap()(matte)
            else:
                trimap_path = os.path.join(self.trimap_dir, img_name).replace('.jpg', '.png')
                trimap = Image.open(trimap_path).convert('L')

            sample['img'], sample['trimap'], sample['matte'] = self.trans([img, trimap, matte])

            # get 3-channels trimap.
            trimap_3 = sample['trimap'].repeat(3, 1, 1)
            trimap_3[0, :, :] = (trimap_3[0, :, :] <= 0.1).float()
            trimap_3[1, :, :] = torch.logical_and(trimap_3[1, :, :] < 0.9, trimap_3[1, :, :] > 0.1).float()
            trimap_3[2, :, :] = (trimap_3[2, :, :] >= 0.9).float()

            sample['trimap_3'] = trimap_3

            return sample

        sample['img'] = self.trans([img])
        return sample

    def __len__(self):
        return len(self.img_names)


class TrainDataset(BaseDataset):
    def __init__(self, args):
        self.mode = args.mode
        self.patch_size = args.patch_size
        super(TrainDataset, self).__init__(
            img_dir=args.img,
            trimap_dir=args.trimap,
            matte_dir=args.matte,
            trans=self.__create_transforms(),
            random_trimap=args.random_trimap,
            sample_size=args.sample
        )

    def __create_transforms(self):
        if self.mode == 'end2end':
            return transforms.Compose([
                transforms.RandomCrop(400),
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor()
            ])

        if self.mode == 't-net':
            return transforms.Compose([
                transforms.RandomCrop(400),
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        if self.mode == 'm-net':
            return transforms.Compose([
                transforms.RandomCrop(320),
                transforms.Resize((self.patch_size, self.patch_size)),
                transforms.ToTensor()
            ])


class ValDataset(BaseDataset):
    def __init__(self, args):
        self.mode = args.mode
        self.patch_size = args.patch_size
        super(ValDataset, self).__init__(
            img_dir=args.val_img,
            trimap_dir=args.val_trimap,
            matte_dir=args.val_matte,
            trans=self.__create_transforms(),
            random_trimap=args.random_trimap
        )

    def __create_transforms(self):
        return transforms.Compose([
            # transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor()
        ])


class TestDataset(BaseDataset):
    def __init__(self, args):
        self.mode = args.mode
        self.patch_size = args.patch_size
        super(TestDataset, self).__init__(
            img_dir=args.img,
            trimap_dir=args.trimap,
            matte_dir=args.matte,
            trans=self.__create_transforms(),
            random_trimap=args.random_trimap
        )

    def __create_transforms(self):
        return transforms.Compose([
            transforms.ResizeIfBiggerThan(self.patch_size),
            transforms.ToTensor()
        ])
