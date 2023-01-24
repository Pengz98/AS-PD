import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.sample_strategies import random_sampling, index_points
# import pytorch3d.loss.chamfer as chamfer
# from models.pointnet_autoencoder import ChamferLoss
from utils.loss_functions import SimplifyLoss, ReconstrLoss
from utils.pointnet2_utils import PointNetSetAbstraction
import torch.nn.init as init


from utils.neighbor_search import SoftProjection, query_knn_point
from utils.sample_strategies import fps_sampling, random_sampling

from time import time

cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_chamfer = ReconstrLoss(mean=True).to(cur_device)


# def to_categorical(y, num_classes=40):
#     '''
#     1-hot encodes a tensor
#     y: [B,]
#     num_classes: [1,]
#     return: new_y: [B, num_classes]
#     '''
#     y = y.view(y.shape[0])
#     new_y = torch.eye(num_classes)[y.cpu().data.numpy()]
#     if(y.is_cuda):
#         return new_y.cuda()
#     return new_y


class SELayer(nn.Module):
    def __init__(self, feat_channel, add_channel, reduction=4):
        super(SELayer, self).__init__()
        channel = feat_channel + add_channel
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, feat_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, scale_vec):
        b, c, _ = x.size()
        concat = torch.cat([x, scale_vec], 1)
        y = self.avg_pool(concat).view(b, -1)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)

class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x

class DGCNN_feat(nn.Module):
    def __init__(self, k=40, emb_dims=1024):
        super(DGCNN_feat, self).__init__()
        self.k = k
        self.transform_net = Transform_Net()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        # self.bn8 = nn.BatchNorm1d(256)
        # self.bn9 = nn.BatchNorm1d(256)
        # self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        # self.conv7 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp1 = nn.Dropout(p=dropout)
        # self.conv8 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #                            self.bn9,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp2 = nn.Dropout(p=dropout)
        # self.conv9 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                             self.bn10,
        #                             nn.LeakyReLU(negative_slope=0.2))
        # self.conv10 = nn.Conv1d(128, 3, kernel_size=1, bias=False)

    def forward(self, x, k=0):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=k if k else self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=k if k else self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=k if k else self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=k if k else self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, emb_dims, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, emb_dims+64*3, num_points)

        # x = self.conv7(x)  # (batch_size, emb_dims+64*3, num_points) -> (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv8(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        # x = self.dp2(x)
        # x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        # x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class get_model(nn.Module):
    def __init__(self, use_atten=True, pre_sample='fps'):
        super(get_model, self).__init__()

        self.conva1 = torch.nn.Conv1d(3, 64, 1)
        self.conva2 = torch.nn.Conv1d(64, 64, 1)
        self.conva3 = torch.nn.Conv1d(64, 128, 1)
        self.conva4 = torch.nn.Conv1d(128, 1024, 1)
        self.bna1 = nn.BatchNorm1d(64)
        self.bna2 = nn.BatchNorm1d(64)
        self.bna3 = nn.BatchNorm1d(128)
        self.bna4 = nn.BatchNorm1d(1024)

        self.dgcnn = DGCNN_feat()

        self.sa1 = PointNetSetAbstraction(radius=0.2, nsample=7, in_channel=3+1024+64*3, mlp=[64, 128, 128],
                                          group_all=False)
        self.SELayer = SELayer(feat_channel=128, add_channel=128, reduction=4)

        self.convs1 = torch.nn.Conv1d(3+64*3+1024, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, 3, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

        self.dropouts = nn.Dropout(p=0.4)

        self.use_atten = use_atten

        self.pre_sample = pre_sample


    def forward(self, xyz, num_sample, k_neighbors=0):
        # pointfeat, globalfeat, trans, trans_feat = self.global_feat(xyz)

        if self.pre_sample=='fps':
            # source_subset, sample_idx = fps_sampling(xyz.transpose(2,1), num_sample=num_sample, seed=None if self.training else 0)
            source_subset, sample_idx = fps_sampling(xyz.transpose(2,1), num_sample=num_sample, seed=None)
        else:
            source_subset, sample_idx = random_sampling(xyz.transpose(2,1), num_sample=num_sample)
        # t1 = time()

        # feature embedding via PointNet-vanilla
        # x = F.relu(self.bna1(self.conva1(xyz)))
        # x = F.relu(self.bna2(self.conva2(x)))
        # point_feat = x
        # x = F.relu(self.bna3(self.conva3(x)))
        # x = self.bna4(self.conva4(x))  # [B, D, N]
        # x = torch.max(x, 2, keepdim=True)[0]
        # global_feat = x
        # pointfeat = torch.cat([point_feat, global_feat.view(-1, 1024, 1).repeat(1, 1, xyz.shape[2])], 1)

        # feature embedding via DGCNN
        pointfeat = self.dgcnn(xyz, k_neighbors) # [B, emb_dims+64*3, num_point]

        B,C,N = xyz.shape

        new_xyz = index_points(xyz.transpose(2,1), sample_idx).transpose(2,1)
        new_pointfeat = index_points(pointfeat.transpose(2,1), sample_idx).transpose(2,1)
        S = new_xyz.shape[2]

        concat = torch.cat([new_xyz, new_pointfeat], 1)

        net = self.dropouts(F.leaky_relu(self.bns1(self.convs1(concat))))
        net = self.dropouts(F.leaky_relu(self.bns2(self.convs2(net))))
        net = self.dropouts(F.leaky_relu(self.bns3(self.convs3(net))))

        if self.use_atten:
            _, local_feat = self.sa1(new_xyz, new_xyz, new_pointfeat)
            # res = torch.cat([local_feat, new_pointfeat], 1)     # res link for attention module
            net = self.SELayer(net, local_feat)

        net = self.convs4(net)

        # t2 = time()
        # print('network cost time(ms):', 1000*(t2-t1)/xyz.shape[0])

        net = net.transpose(2, 1).contiguous()
        net = net.view(B, S, 3)

        net = net + new_xyz.transpose(2,1)

        return net, source_subset


class get_sample_loss(nn.Module):
    def __init__(self, alpha, beta, S_beta, S_gamma, S_delta):
        super(get_sample_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.simplify_loss = SimplifyLoss(mean=True, beta=S_beta, gamma=S_gamma, delta=S_delta)

    def forward(self, xyz=None, new_xyz=None, offset=None):

        similarity_loss = self.simplify_loss(new_xyz, xyz)

        offset_loss = torch.mean(torch.norm(offset, dim=-1))

        sample_loss = self.alpha*similarity_loss + self.beta*offset_loss

        return sample_loss


if __name__ == '__main__':
    xyz = torch.randn((4,2048,3)).to('cuda')
    n_sample = 64
    model = get_model().to('cuda')
    x = model(xyz.transpose(2,1), n_sample)
    print(x.shape)
