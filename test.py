import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime

from models import load_model
from dataset import SatelliteDataset
from utils import transform, rle_encode, rle_decode

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-m', '--model', type=str, default="Unet",
        choices=["Unet", "Unet++", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3+"])
    args = parser.parse_args()

    print("options:", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model initialization
    model = load_model(args.model)
    model.to(device)

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            masks = torch.sigmoid(outputs).cpu().numpy()
            # print(images.shape, outputs.shape, masks.shape)
            # masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
            
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result

    submit.to_csv('./submit.csv', index=False)  