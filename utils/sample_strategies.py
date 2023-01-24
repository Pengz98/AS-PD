import torch
import numpy as np
from utils.neighbor_search import query_knn_point
from utils.visualization import plot_3d_point_cloud


def sample_idx_reverse(sample_idx, length):
    device = sample_idx.device
    reverse_sample_idx = []
    for b in range(sample_idx.shape[0]):
        elements, counts = torch.unique(sample_idx[b], return_counts=True)
        elements_npy = elements.cpu().data.numpy()
        cur_reverse_sample_idx = np.delete(np.arange(length), elements_npy).reshape(1,-1)
        reverse_sample_idx.append(cur_reverse_sample_idx)

    reverse_sample_idx = np.concatenate(reverse_sample_idx, 0)
    reverse_sample_idx = torch.from_numpy(reverse_sample_idx).to(device)
    return reverse_sample_idx


def index_points(points, idx):
    '''
    :param points: [B,N,C]
    :param idx: [B,S]
    :return: indexed points: [B,S,C]
    '''
    device = points.device
    B = points.shape[0]

    # if idx.dim() != 2:
    #     idx = idx.squeeze()

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)    # view_shape=[B,1...1], [B,1] typically
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1     # repeat_shape=[1,S] typically
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def fps_centroids_idx(xyz, n_sample, seed=None):
    '''
    :return: sample point index, [B, n_point]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids_idx = torch.zeros(B, n_sample, dtype=torch.long).to(device)        # [B,M]
    distance = torch.ones(B, N).to(device) * 1e10
    if seed is None:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(seed, seed+1, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(n_sample):
        centroids_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)     # [B,N]
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]       # [B,]
    return centroids_idx

def fps_centroids_idx_complete(xyz, n_sample, sampled_idx):
    '''
    :param xyz: [B,N,3]
    :param n_sample: [M,]
    :param sampled_idx: [B,S] (S<M)
    :param seed: int or None
    :return: centroids_idx: [B,M]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    n_sampled = sampled_idx.shape[1]
    centroids_idx = torch.zeros(B, n_sample, dtype=torch.long).to(device)  # [B,M]
    centroids_idx[:, :n_sampled] = sampled_idx
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    # distance initialization
    for i in range(n_sampled):
        farthest = sampled_idx[:,i]
        centroids_idx[:,i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()

    farthest = sampled_idx[:,-1]
    for i in range(n_sampled, n_sample):
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # [B,N]
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]  # [B,]
        centroids_idx[:, i] = farthest
    return centroids_idx


def fps_sampling(points, num_sample, seed=None):

    sample_idx = fps_centroids_idx(points, num_sample, seed=seed)

    sampled_points = index_points(points, sample_idx)

    return sampled_points, sample_idx

def fps_sampling_complete(points, num_sample, sampled_idx):
    if num_sample > sampled_idx.shape[1]:
        sample_idx = fps_centroids_idx_complete(points, num_sample, sampled_idx)
        sampled_points = index_points(points, sample_idx)
    else:
        sampled_pc = index_points(points, sampled_idx)
        sampled_points, local_sample_idx = fps_sampling(sampled_pc, num_sample)
        sample_idx = index_points(sampled_idx.reshape(points.shape[0], -1, 1), local_sample_idx)

    return sampled_points, sample_idx



def random_sampling(points, num_sample, seed=None):
    batch_size, num_point = points.shape[0], points.shape[1]
    sample_idx_list = []
    for b in range(batch_size):
        batch_sample_idx = np.random.choice(num_point, (1, num_sample), replace=False)
        sample_idx_list.append(batch_sample_idx)
    sample_idx = np.concatenate(sample_idx_list, 0)
    sampled_points = index_points(points, sample_idx)
    return sampled_points, sample_idx



def sample_idx_clean(idx, points):
    sample_idx = torch.zeros((1, idx.shape[1])).long()

    for b in range(idx.shape[0]):
        batch_points = points[b, :, :]
        batch_idx = idx[b, :]
        elements, counts = torch.unique(batch_idx, return_counts=True)

        normal_sample_idx = elements[counts == 1]
        repeat_sample_idx = elements[counts != 1]

        if repeat_sample_idx.shape[0] == 0:
            batch_sample_idx = normal_sample_idx
            sample_idx = torch.cat([sample_idx, batch_sample_idx.reshape(1, -1)], 0)
            continue

        tmp_idx = torch.ones(points.shape[1], )
        tmp_idx[batch_idx] = 0
        remain_idx = torch.arange(points.shape[1])[tmp_idx == 1]

        num_need = batch_idx.shape[0] - normal_sample_idx.shape[0] - repeat_sample_idx.shape[0]
        K = 1 + num_need // repeat_sample_idx.shape[0]

        repeat_knn_idx = query_knn_point(nsample=K, xyz=batch_points[remain_idx, :].reshape(1, -1, 3),
                                         new_xyz=batch_points[repeat_sample_idx, :].reshape(1, -1, 3))
        repeat_knn_idx = repeat_knn_idx.cpu()

        for k in range(K):
            cur_knn_idx = repeat_knn_idx[0, :, k]
            if k == 0:
                supp_idx = cur_knn_idx
            else:
                supp_idx = torch.cat([supp_idx, cur_knn_idx])
            if supp_idx.shape[0] >= num_need:
                supp_idx = supp_idx[:num_need]
                break

        batch_sample_idx = torch.cat([normal_sample_idx, repeat_sample_idx, supp_idx], 0)

        sample_idx = torch.cat([sample_idx, batch_sample_idx.reshape(1, -1)], 0)

    return sample_idx[1:, :]


def sample_idx_check(idx):
    cnt = 0
    for b in range(idx.shape[0]):
        elements, counts = torch.unique(idx[b], return_counts=True)
        cnt += sum(counts != 1)

    return cnt


def post_process_block(points, gen_pc):
    idx = query_knn_point(nsample=1, xyz=points, new_xyz=gen_pc)
    sample_idx = idx.cpu().squeeze()

    cnt = 1e10

    while cnt>=10:
        sample_idx = sample_idx_clean(sample_idx, points)
        cnt = sample_idx_check(sample_idx)
    return sample_idx


def get_complementary_points(pcloud, idx):
    dim_num = len(pcloud.shape)
    n = pcloud.shape[dim_num - 2]
    k = idx.shape[dim_num - 2]

    if dim_num == 2:
        comp_idx = get_complementary_idx(idx, n)
        comp_points = pcloud[comp_idx, :]
    else:
        n_example = pcloud.shape[0]
        comp_points = np.zeros([n_example, n - k, pcloud.shape[2]])
        comp_idx = np.zeros([n_example, n - k], dtype=int)

        for i in range(n_example):
            comp_idx[i, :] = get_complementary_idx(idx[i, :], n)
            comp_points[i, :, :] = pcloud[i, comp_idx[i, :], :]

    return comp_points, comp_idx


def get_complementary_idx(idx, n):
    range_n = np.arange(n, dtype=int)
    comp_indicator = np.full(n, True)

    comp_indicator[idx] = False
    comp_idx = range_n[comp_indicator]

    return comp_idx



if __name__ == '__main__':
    pc = torch.rand((4,256,3)) # 4x256x3

    sample_pc, idx = fps_sampling(pc, num_sample=32)

    fps_idx = fps_centroids_idx_complete(pc, 64, idx)

    print(idx.shape)
    print(sample_pc.shape)