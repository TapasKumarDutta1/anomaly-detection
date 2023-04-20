import glob
import os
import sys
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)
import random
import numpy as np
import torch


class dataset_multi(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.num_domains = opts.num_domains
        self.input_dim = opts.input_dim
        self.template = opts.template
        domains = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        if self.template == None:
            self.images = [[]]
        else:
            self.images = [[], [self.template]]
        stats = ""
        for i in glob.glob(self.dataroot + "/*"):
            self.images[0].append(i)
        self.dataset_size = len(self.images[0])

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        if opts.phase == "train":
            transforms.append(RandomCrop(opts.crop_size))
        else:
            transforms.append(CenterCrop(opts.crop_size))
        if not opts.no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        return

    def __getitem__(self, index):
        if self.template == None:
            cls = 0
        else:
            cls = random.randint(0, 10)
            if cls < 9:
                cls = 0
            else:
                cls = 1
        c_org = np.zeros((self.num_domains,))
        data = self.load_img(
            self.images[cls][random.randint(0, len(self.images[cls]) - 1)],
            self.input_dim,
        )
        c_org[cls] = 1
        return data, torch.FloatTensor(c_org)

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert("RGB")
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.dataset_size


class dataset_multi_from_txt(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.num_domains = opts.num_domains
        self.input_dim = opts.input_dim

        domains = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        self.images = [None] * self.num_domains
        stats = ""
        for i in range(self.num_domains):
            txt_file = Path(self.dataroot) / f"{opts.phase}{domains[i]}.txt"
            self.images[i] = txt_file.read_text().split("\n")[:-1]
            stats += "{}: {}".format(domains[i], len(self.images[i]))
        stats += " images"
        self.dataset_size = max([len(self.images[i]) for i in range(self.num_domains)])

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        if opts.phase == "train":
            transforms.append(RandomCrop(opts.crop_size))
        else:
            transforms.append(CenterCrop(opts.crop_size))
        if not opts.no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        return

    def __getitem__(self, index):
        cls = random.randint(0, self.num_domains - 1)
        c_org = np.zeros((self.num_domains,))
        data = self.load_img(
            self.images[cls][random.randint(0, len(self.images[cls]) - 1)],
            self.input_dim,
        )
        c_org[cls] = 1
        return data, torch.FloatTensor(c_org)

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert("RGB")
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.dataset_size


class dataset_single(data.Dataset):
    def __init__(self, opts, domain):
        self.dataroot = opts.dataroot
        domains = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        images = os.listdir(os.path.join(self.dataroot, opts.phase + domains[domain]))
        self.img1 = images
        self.img = [
            os.path.join(self.dataroot, opts.phase + domains[domain], x) for x in images
        ]
        self.size = len(self.img)
        self.input_dim = opts.input_dim

        self.c_org = np.zeros((opts.num_domains,))
        self.c_org[domain] = 1
        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        transforms.append(CenterCrop(opts.crop_size))
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        return

    def image(self):
        return self.img1

    def __getitem__(self, index):
        data = self.load_img(self.img[index], self.input_dim)
        return data

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert("RGB")
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img, self.c_org

    def __len__(self):
        return self.size
