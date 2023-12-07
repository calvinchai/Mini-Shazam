import torch
from torch import nn
from collections import OrderedDict
from torchvision import transforms
import numpy as np
import librosa


class CQTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 32, kernel_size=(12, 3),
             dilation=(1, 1), padding=(6, 0), bias=False)),
            ('norm0', nn.BatchNorm2d(32)), ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(32, 64, kernel_size=(
                13, 3), dilation=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv2', nn.Conv2d(64, 64, kernel_size=(
                13, 3), dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, kernel_size=(
                3, 3), dilation=(1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)), ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv4', nn.Conv2d(64, 128, kernel_size=(
                3, 3), dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)), ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(128, 128, kernel_size=(
                3, 3), dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)), ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv6', nn.Conv2d(128, 256, kernel_size=(
                3, 3), dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)), ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(256, 256, kernel_size=(
                3, 3), dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)), ('relu7', nn.ReLU(inplace=True)),
            ('pool7', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv8', nn.Conv2d(256, 512, kernel_size=(
                3, 3), dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)), ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(512, 512, kernel_size=(
                3, 3), dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)), ('relu9', nn.ReLU(inplace=True)),
        ]))
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc0 = nn.Linear(512, 300)
        self.fc1 = nn.Linear(300, 10000)

    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        N = x.size()[0]
        x = self.features(x)  # [N, 512, 57, 2~15]
        x = self.pool(x)
        x = x.view(N, -1)
        feature = self.fc0(x)
        # x = self.fc1(feature)
        return feature


def cut_data(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = np.random.randint(max_offset)
            data = data[offset:(out_length+offset), :]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0, offset), (0, 0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0, offset), (0, 0)), "constant")
    return data


def cut_data_front(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = 0
            data = data[offset:(out_length+offset), :]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0, offset), (0, 0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0, offset), (0, 0)), "constant")
    return data


def get_feature(model, data):
    data = transform_test(data)
    data = data.unsqueeze(0)
    data = data.to('cuda')
    feature = model(data)

    feature = feature.cpu().detach().numpy()
    return feature


def CQT(in_path, out_path=None):
    """
    Compute the Constant-Q Transform (CQT) of an audio file.

    Args:
    in_path (str): Path to the input audio file.
    out_path (str, optional): Path to save the output. If None, the function returns the CQT data.

    Returns:
    np.ndarray or None: The computed CQT data if out_path is None, otherwise None.
    """
    try:
        data, sr = librosa.load(in_path)
        if len(data) < 1000:
            print(f'File {in_path} is too short.')
            return

        cqt = np.abs(librosa.cqt(y=data, sr=sr))
        mean_size = 20
        height, length = cqt.shape

        # remove the last few frames to make it evenly divided
        cqt = cqt[:, :length - length % mean_size]

        # Vectorized computation for mean
        new_cqt = cqt.reshape(height, -1, mean_size).mean(axis=2)

        if out_path:
            np.save(out_path, new_cqt)
        else:
            return new_cqt

    except Exception as e:
        print(f'Error processing {in_path}: {e}')


transform_test = transforms.Compose([
    lambda x: x.T,
    # lambda x : x-np.mean(x),
    lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
    lambda x: cut_data_front(x, None),
    lambda x: torch.from_numpy(x),
    lambda x: x.permute(1, 0).unsqueeze(0),
])


model = CQTNet()
model.load_state_dict(torch.load(r"checkpoints/CQTNet.pth"))
model.eval()
model.to('cuda')


def main(input_wav):
    data = CQT(input_wav, out_path=None)  # To return data
    feature = get_feature(model, data)  # To return feature
    return feature


if __name__ == '__main__':
    # extract feature for all songs in test set
    



    # input_wav = r"038896.new.wav"
    # input_wav_2 = r"E:\cs682\data\fma_small\038\038896.mp3"
    
    # feature_2 = main(input_wav_2)
    # feature = main(input_wav)
    # print(nn.CosineSimilarity(dim=1, eps=1e-6)(torch.from_numpy(feature), torch.from_numpy(feature_2)))
