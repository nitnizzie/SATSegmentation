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
    parser.add_argument('-m', '--model', type=str, default="Unet",
        choices=["Unet", "Unet++", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3+"])
    parser.add_argument('--model_dir_1', type=str, default=None)
    parser.add_argument('--model_dir_2', type=str, default=None)
    args = parser.parse_args()

    time = datetime.now().strftime('%m_%d_%H:%M:%S')
    model_dir_1 = f'./models/{args.model_dir_1}'
    model_dir_2 = f'./models/{args.model_dir_2}'

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
    model_1, _ = load_model(args.model)
    if args.model_dir_1:
        model_1.load_state_dict(torch.load(model_dir_1))
    model_1.to(device)
    model_2, _ = load_model(args.model)
    if args.model_dir_2:
        model_2.load_state_dict(torch.load(model_dir_2))
    model_2.to(device)

    with torch.no_grad():
        model_1.eval()
        model_2.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            
            outputs_1 = model_1(images)
            outputs_2 = model_2(images)
        
            masks_1 = torch.argmax(outputs_1, dim=1).cpu().numpy()
            masks_2 = torch.argmax(outputs_2, dim=1).cpu().numpy()
            
            # avg
            masks = (masks_1 + masks_2) / 2

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result

    if args.model_dir:
        submit.to_csv(f'./submit/{args.model_dir}_{time}.csv', index=False)
    else:
        submit.to_csv(f'./submit/{args.model}_{time}.csv', index=False)