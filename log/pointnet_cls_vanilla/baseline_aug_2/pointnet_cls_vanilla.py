import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_utils import STNkd, PointNetEncoder, feature_transform_reguliarzer
import numpy as np

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=False, return_global_feat=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.return_global_feat = return_global_feat

        self.convs1 = torch.nn.Conv1d(channel, 64, 1)
        self.convs2 = torch.nn.Conv1d(64, 64, 1)
        self.convs3 = torch.nn.Conv1d(64, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, 1024, 1)
        self.bns1 = nn.BatchNorm1d(64)
        self.bns2 = nn.BatchNorm1d(64)
        self.bns3 = nn.BatchNorm1d(128)
        self.bns4 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bns1(self.convs1(x)))
        x = F.relu(self.bns2(self.convs2(x)))
        x = F.relu(self.bns3(self.convs3(x)))

        x = self.bns4(self.convs4(x))  # [B, D, N]
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        global_feat = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        if self.return_global_feat:
            return x, global_feat
        else:
            return x, None

class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        loss = F.nll_loss(pred, target)
        return loss
