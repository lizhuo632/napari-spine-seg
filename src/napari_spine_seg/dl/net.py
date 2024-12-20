import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(DoubleConv, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 3, padding_mode='reflect', padding=1,bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.Dropout2d(0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, 3, padding_mode='reflect', padding=1,bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.Dropout2d(0.2),
        #     nn.ReLU(inplace=True)
        # )
        if dropout:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    padding_mode="reflect",
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(0.2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    padding_mode="reflect",
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout2d(0.2),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    padding_mode="reflect",
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                # nn.Dropout2d(0.2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    padding_mode="reflect",
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                # nn.Dropout2d(0.2),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512, 1024],
        logits=True,
        dropout=False,
    ):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, features[0], dropout)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = DoubleConv(features[0], features[1], dropout)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DoubleConv(features[1], features[2], dropout)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = DoubleConv(features[2], features[3], dropout)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = DoubleConv(features[3], features[4], dropout)

        self.up1 = nn.ConvTranspose2d(
            features[4], features[3], kernel_size=2, stride=2
        )
        self.conv6 = DoubleConv(features[4], features[3], dropout)

        self.up2 = nn.ConvTranspose2d(
            features[3], features[2], kernel_size=2, stride=2
        )
        self.conv7 = DoubleConv(features[3], features[2], dropout)

        self.up3 = nn.ConvTranspose2d(
            features[2], features[1], kernel_size=2, stride=2
        )
        self.conv8 = DoubleConv(features[2], features[1], dropout)

        self.up4 = nn.ConvTranspose2d(
            features[1], features[0], kernel_size=2, stride=2
        )
        self.conv9 = DoubleConv(features[1], features[0], dropout)

        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.logits = logits

    def forward(self, x):
        c1 = self.conv1(x)
        d1 = self.down1(c1)

        c2 = self.conv2(d1)
        d2 = self.down2(c2)

        c3 = self.conv3(d2)
        d3 = self.down3(c3)

        c4 = self.conv4(d3)
        d4 = self.down4(c4)

        c5 = self.conv5(d4)

        u1 = self.up1(c5)
        merge1 = torch.cat([u1, c4], dim=1)
        c6 = self.conv6(merge1)

        u2 = self.up2(c6)
        merge2 = torch.cat([u2, c3], dim=1)
        c7 = self.conv7(merge2)

        u3 = self.up3(c7)
        merge3 = torch.cat([u3, c2], dim=1)
        c8 = self.conv8(merge3)

        u4 = self.up4(c8)
        merge4 = torch.cat([u4, c1], dim=1)
        c9 = self.conv9(merge4)

        out = self.out(c9)
        if self.logits:
            return out
        else:
            return out.sigmoid()
        # return out


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Shortcut, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2, self).__init__()

        self.cov1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding_mode="reflect",
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.cov2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding_mode="reflect",
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
        self.act2 = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.cov1(x)))
        x = self.bn2(self.cov2(x))
        return self.act2(x + shortcut)


class ResUNet(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]
    ):
        super(ResUNet, self).__init__()
        self.conv1 = DoubleConv2(in_channels, features[0])
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = DoubleConv2(features[0], features[1])
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = DoubleConv2(features[1], features[2])
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = DoubleConv2(features[2], features[3])
        # self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv5 = DoubleConv(features[3], features[4])

        # self.up1 = nn.ConvTranspose2d(
        #     features[4], features[3], kernel_size=2, stride=2)
        # self.conv6 = DoubleConv(features[4], features[3])

        self.up2 = nn.ConvTranspose2d(
            features[3], features[2], kernel_size=2, stride=2
        )
        self.conv7 = DoubleConv(features[3], features[2])

        self.up3 = nn.ConvTranspose2d(
            features[2], features[1], kernel_size=2, stride=2
        )
        self.conv8 = DoubleConv(features[2], features[1])

        self.up4 = nn.ConvTranspose2d(
            features[1], features[0], kernel_size=2, stride=2
        )
        self.conv9 = DoubleConv(features[1], features[0])

        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        d1 = self.down1(c1)

        c2 = self.conv2(d1)
        d2 = self.down2(c2)

        c3 = self.conv3(d2)
        d3 = self.down3(c3)

        c4 = self.conv4(d3)
        # d4 = self.down4(c4)

        # c5 = self.conv5(d4)

        # u1 = self.up1(c5)
        # merge1 = torch.cat([u1, c4], dim=1)
        # c6 = self.conv6(merge1)

        u2 = self.up2(c4)
        merge2 = torch.cat([u2, c3], dim=1)
        c7 = self.conv7(merge2)

        u3 = self.up3(c7)
        merge3 = torch.cat([u3, c2], dim=1)
        c8 = self.conv8(merge3)

        u4 = self.up4(c8)
        merge4 = torch.cat([u4, c1], dim=1)
        c9 = self.conv9(merge4)

        out = self.out(c9)

        return out.sigmoid()
        # return out


if __name__ == "__main__":

    x = torch.randn((2, 1, 256, 256))
    model = UNet()
    print(model(x).shape)


#
