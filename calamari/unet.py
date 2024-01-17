import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        N = 4
        self.up = nn.Upsample(scale_factor=N, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // N)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

    def forward_old(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_Decoder(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet_Decoder, self).__init__()
        # self.l1 = nn.Linear(2048, 2048)
        # nn.init.kaiming_normal_(self.l1.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # self.up1 = Up(2048, 1024, bilinear)
        # self.up2 = Up(1024, 512, bilinear)
        # self.up3 = Up(512, 256, bilinear)
        # self.up4 = Up(256, 128, bilinear)
        # self.up5 = Up(128, 64, bilinear)
        # self.up6 = Up(64, 32, bilinear)
        # self.up7 = Up(32, 16, bilinear)
        # self.up8 = Up(16, 8, bilinear)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.up4 = Up(32, 8, bilinear)

        # self.up1 = Up(32, 16, bilinear)
        # self.up2 = Up(16, 8, bilinear)
        # self.up3 = Up(8, 4, bilinear)
        # self.up4 = Up(4, 2, bilinear)
        # self.up5 = Up(2, 2, bilinear)
        self.outc = OutConv(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.l1(x[:,:,0,0])
        # x = x.reshape(x.shape[0],-1,8,8)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # x = self.up5(x)
        # x = self.up5(x)
        # x = self.up6(x)
        # x = self.up7(x)
        # x = self.up8(x)
        logits = self.sigmoid(self.outc(x))
        # logits = self.outc(x)

        return logits
