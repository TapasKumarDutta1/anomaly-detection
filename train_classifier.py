import argparse
import torch
from torch.nn import functional as F
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
import shutil
import time
from torchvision.models import resnet18
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
import string
import random
from sklearn.metrics import confusion_matrix


# dataset to load images


class MVTecDataset(Dataset):

    """Extract images from specified folder used for training model.

    Args:
        root: path to images
        transform: transformation to apply to the images
        wts: weights associated with each image
    """

    def __init__(
        self,
        root,
        transform,
        wts,
    ):

        # assign image paths to img_path

        self.img_paths = glob.glob(root + "/*.png")

        # stage1 represents images generated using only ok images
        # stage1 represents images generated using ok and template images
        # nok represents anomaly images
        # ok represents non-anomaly images
        # count the number of each type of images and print them
        count = {"stage1": 0, "stage2": 0, "ok": 0, "nok": 0}

        for i in self.img_paths:
            if "stage1" in i:
                count["stage1"] += 1
            elif "stage2" in i:
                count["stage2"] += 1
            elif "nok" in i:
                count["nok"] += 1
            else:
                count["ok"] += 1

        print(count)

        # transformation to apply to images

        self.transform = transform

        # weights to be assigned to images

        self.wts = wts

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # get image path and ground truth for particular index

        img_path = self.img_paths[idx]

        # using weighted

        wt = self.wts[0]

        if "stage1" in img_path:
            wt = self.wts[1]
        elif "stage2" in img_path:
            wt = self.wts[2]

        # read the image and apply transformations

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # create ground truth labels if image contains anomaly set weight to 10
        if "nok" in img_path:
            gt = 1
            wt = 10
        else:
            gt = 0
        return (img, gt, wt)


class STPM(pl.LightningModule):

    """Train Teacher model(ResNet18) to classify anomaly and non-anomaly images from ok, augmented and nok images

    Args:
        dataset_path: path to images
        load_size: size of images
        lr: learning rate of optimizer used
        momentum: momentum of optimizer used
        batch_size: number of images used to calculate gradients
        weight_decay: learning rate decay
    """

    def __init__(
        self,
        dataset_path,
        load_size=768,
        lr=0.4,
        momentum=0.9,
        batch_size=32,
        weight_decay=1e-4,
    ):
        super(STPM, self).__init__()
        self.save_hyperparameters()
        self.load_size = load_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.momentum = momentum
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        # Use pretrained resnet18
        def get_model():
            model = resnet18(pretrained=True)
            model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            model.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            loss_fn = nn.BCELoss(reduce=False)
            return model.to(device), loss_fn

        self.model, self.criterion = get_model()

        # statistics as calculated on imagenet dataset
        self.data_transforms = transforms.Compose(
            [
                transforms.Resize((self.load_size, self.load_size), Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
        )

    # loss is calculated as the weighted sum of samples
    # weights:
    # anomaly: 10
    # non-anomaly: 5
    # non-anomaly-augmented: 1
    def cal_loss(
        self,
        pred,
        y,
        wts,
    ):
        f_loss = torch.sum(wts * self.criterion(pred.float(), y.float()), 0)

        return f_loss

    # configure optimizer

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    # create training dataloader

    def train_dataloader(self):
        image_datasets = MVTecDataset(
            root=self.dataset_path, transform=self.data_transforms, wts=[5, 1, 2]
        )

        train_loader = DataLoader(
            image_datasets, batch_size=self.batch_size, shuffle=True, num_workers=0
        )  # , pin_memory=True)
        return train_loader

    # pass the image through resnet18 and calculate Binarycrossentropy loss

    def training_step(self, batch, batch_idx):

        # get image and weights used for current index

        (x, y, wts) = batch

        # pass the image through teacher and student model

        pred = self.model(x)
        pred = torch.flatten(pred)
        # calculate and return loss

        loss = self.cal_loss(pred, y, wts)
        return loss


def get_args():
    parser = argparse.ArgumentParser(description="ANOMALYDETECTION")
    parser.add_argument("--phase", choices=["train", "test"], default="train")
    parser.add_argument("--dataset_path", default=r"D:\Dataset\mvtec_anomaly_detection")
    parser.add_argument("--category", default="template")
    parser.add_argument("--num_epochs", default=100)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--weight_decay", default=1e-4)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--load_size", default=768)
    parser.add_argument(
        "--project_path",
        default=r"D:\Project_Train_Results\mvtec_anomaly_detection\210624\test",
    )
    parser.add_argument("--amap_mode", choices=["mul", "sum"], default="mul")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # check available devices and assign cuda if available

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))

    # get assigned parameters

    args = get_args()

    # create trainer

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=os.path.join(args.project_path, args.category),
        max_epochs=int(args.num_epochs),
        gpus=[0],
    )

    # create and fit the model

    model = STPM(
        dataset_path=args.dataset_path,
        load_size=args.load_size,
        lr=args.lr,
        momentum=args.momentum,
        batch_size=int(args.batch_size),
        weight_decay=args.weight_decay
    )
    trainer.fit(model)
    save_path = args.project_path
    torch.save(
        {"model_state_dict": model.model.state_dict()},
        save_path + "/classification_weights.pt",
    )
