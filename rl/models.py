import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecEncoder(nn.Module):
    def __init__(self):
        super(SpecEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # Output: [16, 128, 256]
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: [32, 64, 128]
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: [64, 32, 64]
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class SpecDecoder(nn.Module):
    def __init__(self):
        super(SpecDecoder, self).__init__()
        self.convtranspose1 = nn.ConvTranspose2d(64, 32,
                                                 kernel_size=3,
                                                 stride=2,
                                                 padding=1,
                                                 output_padding=1)  # Output: [32, 64, 128]
        self.bn1 = nn.BatchNorm2d(32)
        self.convtranspose2 = nn.ConvTranspose2d(32, 16,
                                                 kernel_size=3,
                                                 stride=2,
                                                 padding=1,
                                                 output_padding=1)  # Output: [16, 128, 256]
        self.bn2 = nn.BatchNorm2d(16)
        self.convtranspose3 = nn.ConvTranspose2d(16, 3,
                                                 kernel_size=3,
                                                 stride=2,
                                                 padding=1,
                                                 output_padding=1)  # Output: [3, 256, 512]

    def forward(self, x):
        x = F.relu(self.bn1(self.convtranspose1(x)))
        x = F.relu(self.bn2(self.convtranspose2(x)))
        x = torch.sigmoid(self.convtranspose3(x))  # Sigmoid to ensure output is between [0, 1]
        return x


class SpecAutoEncoder(nn.Module):
    def __init__(self):
        super(SpecAutoEncoder, self).__init__()
        self.encoder = SpecEncoder()
        self.decoder = SpecDecoder()

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class WavEncoder(nn.Module):
    def __init__(self):
        super(WavEncoder, self).__init__()
        # Adjust the architecture to achieve the desired downsampling
        self.conv1 = nn.Conv1d(1, 8, kernel_size=4, stride=2, padding=1)  # Halving the dimension, doubling channels
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=1)  # Again halving, doubling channels
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1)  # And halving, doubling channels
        self.bn3 = nn.BatchNorm1d(32)
        # Additional pooling layer to reach the desired size
        self.pool = nn.AdaptiveAvgPool1d(2062)  # Adjusted to reach around 1/4th size in total

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x


class WavDecoder(nn.Module):
    def __init__(self):
        super(WavDecoder, self).__init__()
        # Corresponding upsampling layers
        self.convtranspose1 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.convtranspose2 = nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.convtranspose3 = nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1)
        # Upsampling to original size
        self.upsample = nn.Upsample(661500, mode='linear')

    def forward(self, x):
        x = F.relu(self.bn1(self.convtranspose1(x)))
        x = F.relu(self.bn2(self.convtranspose2(x)))
        x = self.convtranspose3(x)
        x = self.upsample(x)
        return x


class WavAutoEncoder(nn.Module):
    def __init__(self):
        super(WavAutoEncoder, self).__init__()
        self.encoder = WavEncoder()
        self.decoder = WavDecoder()

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x







