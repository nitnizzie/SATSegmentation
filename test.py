import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime

from models import load_model
from dataset import SatelliteDataset
from utils import rle_encode, rle_decode
import albumentations as A

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-m', '--model', type=str, default="DeepLabV3",
        choices=["Unet", "Unet++", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3+"])
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--mask_ratio', type=float, default=0.2)
    args = parser.parse_args()

    time = datetime.now().strftime('%m_%d_%H:%M:%S')
    model_dir = f'./models/{args.model_dir}'

    print("options:", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=mean, std=std, always_apply=True),
            A.pytorch.ToTensorV2(),
        ]
    )

    test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model initialization
    model, _ = load_model(args.model)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    if args.model_dir:
        model.load_state_dict(torch.load(model_dir))

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)

            if args.model == 'DeepLabV3':
                outputs = model(images)['out']
                outputs = torch.sigmoid(outputs)
                outputs = outputs.cpu().numpy()
                masks = (outputs > args.mask_ratio).astype(np.uint8).astype(np.float32)

            else:
                outputs = model(images)
                # outputs = torch.argmax(outputs, dim=1)
                # masks = outputs[:, 1, :, :]
                # print(images.shape, outputs.shape, masks.shape)
                # masks = np.squeeze(masks, axis=1)
                print(outputs[0, 1, :, :].view(-1)[:200])
                print(outputs[0, 0, :, :].view(-1)[:200])

                masks = torch.argmax(outputs, dim=1).cpu().numpy()
                # masks = torch.argmin(outputs, dim=1).cpu().numpy()

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result

    if args.model_dir:
        submit.to_csv(f'./submit/{args.model_dir}_{time}_mask_ratio{args.mask_ratio}.csv', index=False)
    else:
        submit.to_csv(f'./submit/{args.model}_{time}_mask_ratio{args.mask_ratio}.csv', index=False)