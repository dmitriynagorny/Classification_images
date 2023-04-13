import torch.nn as nn
import torch.nn.functional as F
import torch
from types import SimpleNamespace


class RNN_1(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(10)

        self.conv1_2 = nn.Conv2d(10, 32, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv2_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 254, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(254)

        self.conv3_2 = nn.Conv2d(254, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv4_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(1024)

        self.dropout = nn.Dropout(p=0.3)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.liner_1 = nn.Linear(1024 * 1 * 1, 128)
        self.liner_2 = nn.Linear(128, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.liner_3 = nn.Linear(64, out)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.bn1(x)

        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.bn2(x)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.bn3(x)

        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.bn4(x)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.bn5(x)

        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.bn6(x)

        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.bn7(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        x = self.liner_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.liner_2(x)
        x = F.relu(x)
        x = self.bn8(x)
        x = self.liner_3(x)

        return x
