import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import load_model
from dataset import SatelliteDataset
from utils import rle_encode, rle_decode

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--lr', type=float, default=0.0001)
    parser.add_argument('-ep', '--epochs', type=int, default=80)
    parser.add_argument('-m', '--model', type=str, default="Unet",
        choices=["Unet", "Unet++", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3+"])
    parser.add_argument('--preprocess_fn', action='store_true', default=False)
    parser.add_argument('--loss_fn', type=str, default='default')

    args = parser.parse_args()

    time = datetime.now().strftime('%m_%d_%H:%M:%S')

    print("options:", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'running on device: {device}')

    # model initialization
    model, preprocess_fn = load_model(args.model)
    model.to(device)

    for named_params in model.named_parameters():
        print(named_params[0], named_params[1].requires_grad)

    # dataset, dataloader 정의
    if args.preprocess_fn:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_deeplab = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=mean, std=std, always_apply=True),
                A.pytorch.ToTensorV2(),
            ]
        )
        dataset = SatelliteDataset(csv_file='./train.csv', transform=transform_deeplab)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    else:
        from utils import transform
        dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)



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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )
    inference_image = None
    inference_mask = None

    # training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):

            images = images.float().to(device)
            masks = masks.float().to(device)

            if inference_mask is None and inference_image is None:
                inference_image = images
                inference_mask = masks

            optimizer.zero_grad()
            if args.model == 'DeepLabV3':
                outputs = model(images)['out']
            else:
                outputs = model(images)


            # outputs = torch.softmax(outputs, dim=1)
            # outputs = torch.argmax(outputs, dim=1)

            masks = F.one_hot(torch.tensor(masks).to(torch.int64), num_classes=2).permute(0, 3, 1, 2).float().to(device)


            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        with torch.no_grad():
            import numpy as np

            plt.imshow(np.array(inference_image[0].cpu().permute(1, 2, 0)))
            plt.show()
            if inference_mask[0].shape[0] == 2:
                plt.imshow(np.array(torch.argmax(inference_mask[0], dim=0).cpu()))
            else:
                plt.imshow(np.array(inference_mask[0].cpu()))
            plt.show()

            if args.model == 'DeepLabV3':
                test_output = model(inference_image)['out'][0]
            else:
                test_output = model(inference_image)[0]
            # plt.imshow(np.array(torch.argmax(test_output, dim=0).cpu()))
            test_output = np.squeeze(np.array(torch.argmax(test_output, dim=0).cpu()))

            # test_output_35 = (test_output > 0.35).astype(np.uint8)  # Threshold = 0.35
            # test_output_50 = (test_output > 0.5).astype(np.uint8)  # Threshold = 0.5
            # test_output_75 = (test_output > 0.75).astype(np.uint8)  # Threshold = 0.75

            plt.imshow(test_output)
            plt.show()
            # plt.imshow(test_output_35)
            # plt.show()
            # plt.imshow(test_output_50)
            # plt.show()
            # plt.imshow(test_output_75)
            # plt.show()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')



    # save model
    model_dir = f'./models/{args.model}_{time}_lossfn{args.loss_fn}_lr{args.lr}_epoch{args.epochs}.pt'
    torch.save(model.state_dict(), model_dir)
    print(f"Model saved at {model_dir}")
