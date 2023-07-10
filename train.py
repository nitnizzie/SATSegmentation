import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from datetime import datetime

from models import load_model
from dataset import SatelliteDataset
from utils import transform, rle_encode, rle_decode

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--lr', type=float, default=0.0001)
    parser.add_argument('-ep', '--epochs', type=int, default=80)
    parser.add_argument('-m', '--model', type=str, default="Unet",
        choices=["Unet", "Unet++", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3+"])
    args = parser.parse_args()

    time = datetime.now().strftime('%m_%d_%H:%M:%S')

    print("options:", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # model initialization
    model = load_model(args.model)
    model.to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')
    
    # save model
    model_dir = f'./models/{args.model}_{time}.pt'
    torch.save(model.state_dict(), model_dir)
    print(f"Model saved at {model_dir}")
