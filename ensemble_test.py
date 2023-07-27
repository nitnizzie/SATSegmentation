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
    parser.add_argument('--mask_ratio', type=float, default=0.35)
    args = parser.parse_args()

    time = datetime.now().strftime('%m_%d_%H:%M:%S')

    if args.model_dir:
        input_list = args.model_dir
        print("List of input strings:", input_list)
        for idx, model_dir in enumerate(input_list):
            input_list[idx] = f'./models/{model_dir}'
    else:
        print("No input strings provided.")

    print("options:", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_list = []
    transform_list.append(A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=mean, std=std, always_apply=True),
            A.pytorch.ToTensorV2(),
        ]
    ))
    transform_list.append(A.Compose(
        [
            A.Resize(224, 224),
            # Rotate 90
            A.Affine(rotate=90, always_apply=True),
            A.Normalize(mean=mean, std=std, always_apply=True),
            A.pytorch.ToTensorV2(),
        ]
    ))
    transform_list.append(A.Compose(
        [
            A.Resize(224, 224),
            # Rotate 180
            A.Affine(rotate=180, always_apply=True),
            A.Normalize(mean=mean, std=std, always_apply=True),
            A.pytorch.ToTensorV2(),
        ]
    ))
    transform_list.append(A.Compose(
        [
            A.Resize(224, 224),
            # Rotate 270
            A.Affine(rotate=270, always_apply=True),
            A.Normalize(mean=mean, std=std, always_apply=True),
            A.pytorch.ToTensorV2(),
        ]
    ))


    test_dataset_list = []
    for transform in transform_list:
        test_dataset_list.append(SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True))
    
    test_dataloader_list = []
    for test_dataset in test_dataset_list:
        test_dataloader_list.append(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4))

    # model initialization
    model, _ = load_model(args.model)
    if args.model_dir:
        model.load_state_dict(torch.load(model_dir))
    model.to(device)

    with torch.no_grad():
        model.eval()
        result=[]
        output_list = []
        for test_dataloader, i in test_dataloader_list:
            partial_result = []
            for images in tqdm(test_dataloader):
                images = images.float().to(device)

                output = model(images)['out']

                # rotate to original
                output = torch.rot90(output, k=4-i, dims=(2, 3))   
                output = torch.sigmoid(output)
                partial_result.append(output)

            output_list.append(partial_result)
        
        avg_output = torch.zeros_like(output_list[0]).to(device)
        for output in output_list:
            avg_output += output
        avg_output /= len(output_list)
        avg_output = avg_output.cpu().numpy()
        masks = (avg_output > args.mask_ratio).astype(np.uint8).astype(np.float32)

        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)
                        
    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result

    if args.model_dir:
        submit.to_csv(f'./submit/{args.model}_{time}_mask_ratio{args.mask_ratio}_ensemble.csv', index=False)
    else:
        submit.to_csv(f'./submit/{args.model}_{time}_mask_ratio{args.mask_ratio}_ensemble.csv', index=False)