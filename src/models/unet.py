import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional Block with two convolutional layers followed by batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        x (Tensor): Output tensor after applying the convolutional block.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    """
    Encoder block that consists of a convolutional block followed by max pooling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        x (Tensor): Output tensor after applying the convolutional block.
        p (Tensor): Pooled tensor after applying max pooling.
    """
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p


class Decoder(nn.Module):
    """
    Decoder block that consists of a transposed convolution followed by a convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        x (Tensor): Output tensor after applying the transposed convolution and concatenation.
    """
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    Baseline U-Net architecture for image segmentation.

    Args:
        num_classes (int): Number of output classes for segmentation.

    Returns:
        out (Tensor): Output tensor after applying the U-Net architecture.
    """
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        # Encoder
        self.encoder1 = Encoder(3, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        # Decoder
        self.decoder4 = Decoder(1024, 512)
        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # Change to 3 for RGB output

    def forward(self, x):
        # Encoder
        enc1, p1 = self.encoder1(x)
        enc2, p2 = self.encoder2(p1)
        enc3, p3 = self.encoder3(p2)
        enc4, p4 = self.encoder4(p3)
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        # Decoder
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        # Final Convolution
        out = self.final_conv(dec1)

        return out
