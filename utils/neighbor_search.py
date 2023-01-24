"""PyTorch implementation of the Soft Projection block."""

import torch
import torch.nn as nn
import numpy as np

from pytorch3d.ops.knn import knn_points
from pointnet2_ops.pointnet2_utils import grouping_operation as group_point

import math

cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def complete_fps_centroids_idx(xyz, n_sample, sampled_idx, seed=None):
    '''
    :return: sample point index, [n_point,]
    '''
    device = xyz.device
    N, C = xyz.shape
    centroids_idx = torch.zeros((n_sample,1), dtype=torch.long).to(device)        # [B,N]
    distance = torch.ones(N).to(device) * 1e10
    if seed is None:
        farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(seed, seed+1, (1,), dtype=torch.long).to(device)

    if sampled_idx is not None:
        centroids_idx[:sampled_idx.shape[0],:] = sampled_idx
        distance[sampled_idx] = 0
        unsampled_idx = torch.where(distance!=0)[0]
        farthest = unsampled_idx[torch.randint(unsampled_idx.shape[0], (1,))]
        start_i = sampled_idx.shape[0]
    else:
        start_i = 0

    for i in range(start_i, n_sample):
        centroids_idx[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)     # [N,]
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.argmax(distance)     # [1,]
    return centroids_idx

def index_points(points, idx):
    '''
    :param points: [B,N,C]
    :param idx: [B,S]
    :return: indexed points: [B,S,C]
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)    # view_shape=[B,1...1], [B,1] typically
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1     # repeat_shape=[1,S] typically
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def fps_complete(sampled_idx, full_pc, n_sample):
    '''
    :param sampled_idx: [Sx1]
    :param full_pc: [Nx3]
    :param n_sample: M (>S)
    :return: sampled_pc_completed [Mx3]
    '''
    completed_idx = complete_fps_centroids_idx(full_pc, n_sample, sampled_idx.reshape(-1,1))
    sampled_pc_completed = index_points(full_pc.reshape(1,-1,3), completed_idx.reshape(1,-1)).squeeze()
    return sampled_pc_completed, completed_idx




def query_knn_point(nsample, xyz, new_xyz):
    dist, idx, nn = knn_points(new_xyz, xyz, K=nsample)
    return idx


def knn_point(group_size, point_cloud, query_cloud):
    # knn_obj = KNN(k=group_size, transpose_mode=False)
    # dist, idx = knn_obj(point_cloud, query_cloud)
    point_cloud = point_cloud.transpose(2, 1)
    query_cloud = query_cloud.transpose(2, 1)
    dist, idx, _ = knn_points(p1=query_cloud, p2=point_cloud, K=group_size)
    dist = dist.transpose(2, 1)
    idx = idx.transpose(2, 1)
    return dist, idx


def _axis_to_dim(axis):
    """Translate Tensorflow 'axis' to corresponding PyTorch 'dim'"""
    return {0: 0, 1: 2, 2: 3, 3: 1}.get(axis)


class SoftProjection(nn.Module):
    def __init__(
        self,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=False,
        all_epoch=200,
        min_sigma=1e-4
    ):
        """Computes a soft nearest neighbor point cloud.
        Arguments:
            group_size: An integer, number of neighbors in nearest neighborhood.
            initial_temperature: A positive real number, initialization constant for temperature parameter.
            is_temperature_trainable: bool.
        Inputs:
            point_cloud: A `Tensor` of shape (batch_size, 3, num_orig_points), database point cloud.
            query_cloud: A `Tensor` of shape (batch_size, 3, num_query_points), query items to project or propogate to.
            point_features [optional]: A `Tensor` of shape (batch_size, num_features, num_orig_points), features to propagate.
            action [optional]: 'project', 'propagate' or 'project_and_propagate'.
        Outputs:
            Depending on 'action':
            propagated_features: A `Tensor` of shape (batch_size, num_features, num_query_points)
            projected_points: A `Tensor` of shape (batch_size, 3, num_query_points)
        """

        super().__init__()
        self._group_size = group_size

        # create temperature variable
        self._temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                requires_grad=is_temperature_trainable,
                dtype=torch.float32,
                device=cur_device
            )
        )
        self.initial_sigma = initial_temperature ** 2
        self.is_temperature_trainable = is_temperature_trainable

        self.anneal_rate = math.exp(math.log(min_sigma/self.initial_sigma)/all_epoch)

        self.all_epoch = all_epoch
        self.min_sigma = torch.tensor(min_sigma, dtype=torch.float32, requires_grad=False)


    def forward(self, point_cloud, query_cloud, epoch=None, point_features=None, action="project"):
        point_cloud = point_cloud.contiguous()
        query_cloud = query_cloud.contiguous()

        if action == "project":
            return self.project(point_cloud, query_cloud, epoch)
        elif action == "propagate":
            return self.propagate(point_cloud, point_features, query_cloud)
        elif action == "project_and_propagate":
            return self.project_and_propagate(point_cloud, point_features, query_cloud)
        else:
            raise ValueError(
                "action should be one of the following: 'project', 'propagate', 'project_and_propagate'"
            )

    def _group_points(self, point_cloud, query_cloud, point_features=None):
        group_size = self._group_size

        # find nearest group_size neighbours in point_cloud
        dist, idx = knn_point(group_size, point_cloud, query_cloud)

        # self._dist = dist.unsqueeze(1).permute(0, 1, 3, 2) ** 2

        idx = idx.permute(0, 2, 1).type(
            torch.int32
        )  # index should be Batch x QueryPoints x K
        grouped_points = group_point(point_cloud, idx)  # B x 3 x QueryPoints x K
        grouped_features = (
            None if point_features is None else group_point(point_features, idx)
        )  # B x F x QueryPoints x K
        return grouped_points, grouped_features

    def _get_distances_anneal(self, grouped_points, query_cloud, epoch):
        deltas = grouped_points - query_cloud.unsqueeze(-1).expand_as(grouped_points)
        dist = torch.sum(deltas ** 2, dim=_axis_to_dim(3), keepdim=True) / self.sigma(epoch)
        return dist

    def _get_distances(self, grouped_points, query_cloud):
        deltas = grouped_points - query_cloud.unsqueeze(-1).expand_as(grouped_points)
        dist = torch.sum(deltas ** 2, dim=_axis_to_dim(3), keepdim=True)
        return dist

    def sigma(self, epoch):
        device = self._temperature.device
        if self.is_temperature_trainable:
            return max(self._temperature ** 2, self.min_sigma.to(device))
        else:
            temperature_2 = self.initial_sigma * self.anneal_rate ** epoch
            temperature_2 = torch.tensor(temperature_2, dtype=torch.float32, device=device)
            return max(temperature_2, self.min_sigma.to(device))

    def project_and_propagate(self, point_cloud, point_features, query_cloud):
        # group into (batch_size, num_query_points, group_size, 3),
        # (batch_size, num_query_points, group_size, num_features)
        grouped_points, grouped_features = self._group_points(
            point_cloud, query_cloud, point_features
        )
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))

        # get weighted average of grouped_points
        projected_points = torch.sum(
            grouped_points * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)
        propagated_features = torch.sum(
            grouped_features * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)

        return (projected_points, propagated_features)

    def propagate(self, point_cloud, point_features, query_cloud):
        grouped_points, grouped_features = self._group_points(
            point_cloud, query_cloud, point_features
        )
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))

        # get weighted average of grouped_points
        propagated_features = torch.sum(
            grouped_features * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)

        return propagated_features

    def project(self, point_cloud, query_cloud, epoch, hard=False):
        grouped_points, _ = self._group_points(point_cloud, query_cloud)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))
        if hard:
            raise NotImplementedError

        # get weighted average of grouped_points
        weights = weights.repeat(1, 3, 1, 1)
        projected_points = torch.sum(
            grouped_points * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)
        return projected_points


"""SoftProjection test"""


if __name__ == "__main__":
    k = 3
    propagator = SoftProjection(k, initial_temperature=1.0)
    query_cloud = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [5, 4, 4],
            [4, 5, 4],
            [4, 4, 5],
            [8, 7, 7],
            [7, 8, 7],
            [7, 7, 8],
        ]
    )
    point_cloud = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 5, 5], [7, 7, 8], [7, 7, 8.5]]
    )
    point_features = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
        ]
    )
    expected_nn_cloud = np.array(
        [
            [0.333, 0.333, 0.333],
            [1, 0, 0],
            [1, 0, 0],
            [4.333, 4.333, 4.333],
            [7, 7, 8],
            [7, 7, 8],
        ]
    )
    expected_features_nn_1 = np.array(
        [
            [6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [16, 17, 18, 19, 20],
            [16, 17, 18, 19, 20],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [21, 22, 23, 24, 25],
            [21, 22, 23, 24, 25],
        ]
    )
    expected_features_nn_3 = np.array(
        [
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [2.459, 3.459, 4.459, 5.459, 6.459],
            [2.459, 3.459, 4.459, 5.459, 6.459],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [22.113, 23.113, 24.113, 25.113, 26.113],
            [22.113, 23.113, 24.113, 25.113, 26.113],
            [23.189, 24.189, 25.189, 26.189, 27.189],
        ]
    )

    if k == 3:
        expected_features_nn = expected_features_nn_3
    elif k == 1:
        expected_features_nn = expected_features_nn_1
    else:
        assert False, "Non valid value of k"

    # expend to batch_size = 1
    point_cloud = np.expand_dims(point_cloud, axis=0)
    point_features = np.expand_dims(point_features, axis=0)
    query_cloud = np.expand_dims(query_cloud, axis=0)
    expected_features_nn = np.transpose(
        np.expand_dims(expected_features_nn, axis=0), (0, 2, 1)
    )
    expected_nn_cloud = np.transpose(
        np.expand_dims(expected_nn_cloud, axis=0), (0, 2, 1)
    )

    point_cloud_pl = (
        torch.tensor(point_cloud, dtype=torch.float32).permute(0, 2, 1).cuda()
    )
    point_features_pl = (
        torch.tensor(point_features, dtype=torch.float32).permute(0, 2, 1).cuda()
    )
    query_cloud_pl = (
        torch.tensor(query_cloud, dtype=torch.float32).permute(0, 2, 1).cuda()
    )

    propagator.cuda()
    # projected_points, propagated_features = propagator.project_and_propagate(point_cloud_pl, point_features_pl, query_cloud_pl)
    propagated_features = propagator.propagate(
        point_cloud_pl, point_features_pl, query_cloud_pl
    )
    propagated_features = propagated_features.cpu().detach().numpy()

    # replace Query and Point roles, reduce temperature:
    state_dict = propagator.state_dict()
    state_dict['_temperature'] = torch.tensor(0.1, dtype=torch.float32)
    propagator.load_state_dict(state_dict)
    projected_points = propagator.project(query_cloud_pl, point_cloud_pl)
    projected_points = projected_points.cpu().detach().numpy()

    print("propagated features:")
    print(propagated_features)

    print("projected points:")
    print(projected_points)

    expected_features_nn = expected_features_nn.squeeze()
    expected_nn_cloud = expected_nn_cloud.squeeze()
    propagated_features = propagated_features.squeeze()
    projected_points = projected_points.squeeze()

    mse_feat = np.mean(
        np.sum((propagated_features - expected_features_nn) ** 2, axis=1)
    )
    mse_points = np.mean(np.sum((projected_points - expected_nn_cloud) ** 2, axis=1))
    print("propagated features vs. expected NN features mse:")
    print(mse_feat)
    print("projected points vs. expected NN points mse:")
    print(mse_points)