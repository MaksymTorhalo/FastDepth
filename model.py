import torch.nn as nn
from torch import randn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn.functional as F




class EncoderMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        mobileNet_V2 = mobilenet_v2(weights=weights)
        features = mobileNet_V2.features            # all feature extraction layers, without the classifier
        self.stage1 = features[0:2]     # res 112, 2nd(last), first skip
        self.stage2 = features[2:4]     # res 56, 2nd(last), second skip
        self.stage3 = features[4:7]     # res 28, 3rd(last), third skip
        self.stage4 = features[7:]      # res 14 -> 7, rest, encoder out

        # print(features)
        # print("\n\n\n\n\n\n-----------------------------------\n\n\n\n\n")
        # x = randn(1, 3, 224, 224)
        # for i, layer in enumerate(features):
        #     x = layer(x)
        #     print(f'Index {i}: Shape {list(x.shape)}')

    def forward(self, x):
        skip1 = self.stage1(x)
        skip2 = self.stage2(skip1)
        skip3 = self.stage3(skip2)
        x = self.stage4(skip3)

        return x, skip1, skip2, skip3


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        '''
        Input and output stay the same since we don't combine them yet in the Depthwise layer.
        Padding should be kernel_size // 2, since half of the kernel will convolute the pixels
        outside the matrix at the edges.
        Groups = in_channels, since we want separate filters applied to every channel, instead of
        grouped stacks of filters.
        '''
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NNConv5(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderMobileNet()

        self.upSample1 = DepthwiseSeparableConv(in_channels=1280, out_channels=512, kernel_size=5)
        self.upSample2 = DepthwiseSeparableConv(in_channels=512, out_channels=256, kernel_size=5)
        self.upSample3 = DepthwiseSeparableConv(in_channels=256, out_channels=128, kernel_size=5)
        self.upSample4 = DepthwiseSeparableConv(in_channels=128, out_channels=64, kernel_size=5)
        self.upSample5 = DepthwiseSeparableConv(in_channels=64, out_channels=32, kernel_size=5)

        self.proj3 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(in_channels=24, out_channels=128, kernel_size=1, bias=False)
        self.proj1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1, bias=False)

        self.pointwise = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, bias=False)

    def forward(self, x):
        x, skip1, skip2, skip3 = self.encoder(x)

        x = self.upSample1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.upSample2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x + self.proj3(skip3)

        x = self.upSample3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x + self.proj2(skip2)

        x = self.upSample4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = x + self.proj1(skip1)

        x = self.upSample5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        return self.pointwise(x)



