import torch
from torch.utils.data import DataLoader
import sys
import logging
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import load_model
from dataset import SatelliteDataset
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn

from utils import rle_encode, rle_decode, save_model

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--lr', type=float, default=0.0001)
    parser.add_argument('-ep', '--epochs', type=int, default=80)
    parser.add_argument('-m', '--model', type=str, default="DeepLabV3",
        choices=["Unet", "Unet++", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3+"])
    parser.add_argument('--preprocess_fn', action='store_true', default=True)
    parser.add_argument('--loss_fn', type=str, default='dice_v2')
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--transform', type=int, default=2)
    parser.add_argument('--wo_sigmoid', action='store_true', default=True)
    parser.add_argument('-d', '--dataset', type=int, default=4)



    args = parser.parse_args()
    time = datetime.now().strftime('%m_%d_%H:%M:%S')

    # file name
    fname = f"{args.model}_{time}_lossfn{args.loss_fn}_lr{args.lr}_epoch{args.epochs}_transform{args.transform}_dataset{args.dataset}"

    if args.wo_sigmoid:
        fname += "_wo_sigmoid"


    # Create a logger
    logger = logging.getLogger("stdout_logger")
    logger.setLevel(logging.INFO)
    log_file = f"log/{fname}.log"
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print("options:", args)

    device = f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu'
    print(f'running on device: {device}')
    logger.info('running on device: %s', device)
    logger.info("options: %s", args)

    # model initialization
    model, preprocess_fn = load_model(args.model)
    model.to(device)

    for named_params in model.named_parameters():
        print(named_params[0], named_params[1].requires_grad)

    # dataset, dataloader 정의
    if args.preprocess_fn:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if args.dataset != 10:
            if args.transform == 0:
                transform_deeplab = A.Compose(
                    [
                        A.Resize(224, 224),
                        A.Normalize(mean=mean, std=std, always_apply=True),
                        A.pytorch.ToTensorV2(),
                    ]
                )
            elif args.transform == 1:
                transform_deeplab = A.Compose(
                    [
                        A.RandomCrop(224, 224),
                        A.Flip(),
                        A.Normalize(mean=mean, std=std, always_apply=True),
                        A.pytorch.ToTensorV2(),
                    ]
                )
            elif args.transform == 2:
                transform_deeplab = A.Compose(
                    [
                        # RandomSizedCrop
                        A.RandomSizedCrop(
                            min_max_height=(224, 224), height=224, width=224, p=1
                        ),
                        A.HorizontalFlip(p=0.5),
                        A.Rotate(limit=[-10, 10], p=0.5),
                        A.Normalize(mean=mean, std=std, always_apply=True),
                        A.pytorch.ToTensorV2(),
                    ]
                )
            else:
                raise NotImplementedError

            dataset_1 = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab, args=args)
            dataset_2 = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab, args=args)
            dataset_3 = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab, args=args)
            dataset_4 = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab, args=args)

            if args.dataset == 2:
                dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2])
            elif args.dataset == 4:
                dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4])
            elif args.dataset == 8:
                dataset_5 = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab, args=args)
                dataset_6 = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab, args=args)
                dataset_7 = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab, args=args)
                dataset_8 = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab, args=args)
                dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8])

        elif args.dataset == 10:
            transform_default = A.Compose(
                [
                    A.RandomSizedCrop(
                        min_max_height=(224, 224), height=224, width=224, p=1
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=[-10, 10], p=0.5),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                    A.pytorch.ToTensorV2(),
                ]
            )
            tranform_blur = A.Compose(
                [
                    A.RandomSizedCrop(
                        min_max_height=(224, 224), height=224, width=224, p=1
                    ),
                    A.AdvancedBlur(blur_limit=3, p=1, always_apply=True),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                    A.pytorch.ToTensorV2(),
                ]
            )
            tranform_gauss = A.Compose(
                [
                    A.RandomSizedCrop(
                        min_max_height=(224, 224), height=224, width=224, p=1
                    ),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1, always_apply=True),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                    A.pytorch.ToTensorV2(),
                ]
            )
            transform_color = A.Compose(
                [
                    A.RandomSizedCrop(
                        min_max_height=(224, 224), height=224, width=224, p=1
                    ),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=True, p=1),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                    A.pytorch.ToTensorV2(),
                ]
            )
            dataset_list = []
            transform_list = [tranform_blur, tranform_gauss, transform_color]
            for i in range(5):
                transform_list.append(transform_default)

            for i in range(len(transform_list)):
                dataset_list.append(
                    SatelliteDataset(csv_file='./train.csv', transform=transform_list[i], args=args)
                )
            dataset = torch.utils.data.ConcatDataset(dataset_list)

        else:
            raise NotImplementedError
        

        dataset_size = len(dataset)

        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4)
    else:
        raise NotImplementedError
        # from utils import transform
        # dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
        # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



    # loss function과 optimizer 정의

    if args.loss_fn == 'default':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss_fn == 'dice':
        from utils import DiceLoss
        criterion = DiceLoss()
    elif args.loss_fn == 'dice_v2':
        from utils import dice_loss
        criterion = dice_loss
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # define learning rate scheduler (not used in this NB)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    # )
    inference_image = None
    inference_mask = None

    lowest_loss_yet = 100000

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    # training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_dataloader):

            if len(images.shape) == 5:
                images = torch.stack(images)
                masks = torch.stack(masks)

            images = images.float().to(device)
            masks = masks.float().to(device)

            if inference_mask is None and inference_image is None:
                inference_image = images
                inference_mask = masks

            optimizer.zero_grad()
            if args.model == 'DeepLabV3':
                if args.wo_sigmoid:
                    outputs = model(images)['out']
                else:
                    outputs = torch.sigmoid(model(images)['out'])
                loss = criterion(outputs.squeeze(), masks.squeeze())
            else:
                outputs = model(images)
                masks = F.one_hot(torch.tensor(masks).to(torch.int64), num_classes=2).permute(0, 3, 1, 2).float().to(
                    device)
                loss = criterion(outputs, masks)


            # outputs = torch.softmax(outputs, dim=1)
            # outputs = torch.argmax(outputs, dim=1)


            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_dice_score = []
            for images, masks in tqdm(val_dataloader):
                if len(images.shape) == 5:
                    images = torch.stack(images)
                    masks = torch.stack(masks)
        
                images = images.float().to(device)
                masks = masks.float().to(device)
        
                if args.model == 'DeepLabV3':
                    outputs = torch.sigmoid(model(images)['out'])
                    val_loss = criterion(outputs.squeeze(), masks.squeeze())
        
                    numpy_outputs = outputs.squeeze().cpu().numpy()
        
                    # cast to uint8
                    mask_05 = (numpy_outputs > 0.5).astype(np.uint8).astype(np.float32)
                    mask_03 = (numpy_outputs > 0.3).astype(np.uint8).astype(np.float32)
                    mask_02 = (numpy_outputs > 0.2).astype(np.uint8).astype(np.float32)
        
                    from utils import dice_score
                    dice_score_05 = dice_score(mask_05, masks.squeeze().cpu().numpy())
                    dice_score_03 = dice_score(mask_03, masks.squeeze().cpu().numpy())
                    dice_score_02 = dice_score(mask_02, masks.squeeze().cpu().numpy())
        
                    val_dice_score.append([dice_score_05, dice_score_03, dice_score_02])
        
                else:
                    outputs = model(images)
                    masks = F.one_hot(torch.tensor(masks).to(torch.int64), num_classes=2).permute(0, 3, 1, 2).float().to(
                        device)
                    val_loss = criterion(outputs, masks)
                    val_loss += val_loss.item()
        
        
            if args.model == 'DeepLabV3':
                val_dice_score = np.mean(val_dice_score, axis=0)
                print(f'val_loss {val_loss / len(val_dataloader)} val_dice_score_05 {val_dice_score[0]}, val_dice_score_03 {val_dice_score[1]}, val_dice_score_02 {val_dice_score[0]}')
                logger.info(f'val_loss {val_loss / len(val_dataloader)} val_dice_score_05 {val_dice_score[0]}, val_dice_score_03 {val_dice_score[1]}, val_dice_score_02 {val_dice_score[0]}')
            else:
                print(f'val_loss {val_loss / len(val_dataloader)}')
                logger.info(f'val_loss {val_loss / len(val_dataloader)}')

        save_model(model, fname)
        print(f'lowest loss, saving current model at epoch: {epoch}')

    save_model(model, fname + '_final')