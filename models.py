import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def load_model(model_name: str) -> nn.Module:
    if model_name == "Unet":
        return UNet()
    # elif model_name == "Unet++":
    #     return UnetPlus()
    # elif model_name == "FPN":
    #     return FPN()
    # elif model_name == "PSPNet":
    #     return PSPNet()
    # elif model_name == "DeepLabV3":
    #     return DeepLabV3()
    elif model_name == "DeepLabV3+":
        # segmentation model - deeplabv3
        ENCODER = 'resnet101'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['buildings', 'background']
        ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

        # deeplab v3
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )

        return model
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")

# U-Net component
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# Simple U-Net
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out