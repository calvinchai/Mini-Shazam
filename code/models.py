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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
