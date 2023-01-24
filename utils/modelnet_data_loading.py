import numpy as np
import h5py
import os
import torch
from torch.utils.data import Dataset

DATA_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = DATA_UTILS_DIR.rsplit('/', 1)[0]
DATA_DIR = os.path.join(BASE_DIR, '../dataset')

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def pc_normalize(pc):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
    pc = pc / m
    return pc


def batch_pc_normalize(pc):
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=2)), dim=1)[0].reshape(-1,1,1)
    pc = pc / m
    return pc, m, centroid


def batch_pc_recover(pc, m, centroid):
    device = pc.device
    m = m.to(device)
    centroid = centroid.to(device)
    pc = pc * m
    pc = pc + centroid
    del m, centroid
    return pc

def load_h5_data_label(filename, base_dir=DATA_DIR):
    data_dir = os.path.join(base_dir, filename)
    f = h5py.File(data_dir)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


class ModelNet40_h5(Dataset):
    def __init__(self, h5_filename, dataset_dir, n_point, class_choice=None, pc_normalize=False, uniform_sample=False):
        if class_choice is not None:
            data, label = load_h5_data_label(h5_filename, base_dir=dataset_dir)
            index = np.where(label==class_choice)[0]
            self.data, self.label = data[index], label[index]
        else:
            self.data, self.label = load_h5_data_label(h5_filename, base_dir=dataset_dir)
        self.n_point = n_point
        self.pc_normalize = pc_normalize
        self.uniform = uniform_sample

    def __getitem__(self, index):
        if self.uniform:
            data = farthest_point_sample(self.data[index], self.n_point)
        else:
            data = self.data[index][:self.n_point, :]

        data = torch.from_numpy(data).float()
        if self.pc_normalize:
            data = pc_normalize(data)
        label = torch.from_numpy(self.label[index]).long().squeeze()
        return data, label

    def __len__(self):
        return len(self.data)


def get_filenames_in_txt(txt_files):
    return [line.rstrip().rsplit('/', 1)[-1] for line in open(txt_files)]


def ModelNet40_h5_dataset_list(dataset_name, split='default', n_point=1024, class_choice=None, normalize=False, uniform_sample=False):
    ''' datasetname: the directory name of used dataset. Asumming that dataset is placed in the 'data/' path'''
    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    dataset_list = []
    if split == 'train':
        train_files_txt = os.path.join(dataset_dir, 'train_files.txt')
        train_filenames = get_filenames_in_txt(train_files_txt)
        filenames = train_filenames
    elif split == 'test':
        test_files_txt = os.path.join(dataset_dir, 'test_files.txt')
        test_filenames = get_filenames_in_txt(test_files_txt)
        filenames = test_filenames
    else:
        train_files_txt = os.path.join(dataset_dir, 'train_files.txt')
        test_files_txt = os.path.join(dataset_dir, 'test_files.txt')
        train_filenames = get_filenames_in_txt(train_files_txt)
        test_filenames = get_filenames_in_txt(test_files_txt)
        filenames = train_filenames + test_filenames
    for i in range(len(filenames)):
        dataset = ModelNet40_h5(filenames[i], dataset_dir=dataset_dir, n_point=n_point, class_choice=class_choice, pc_normalize=normalize, uniform_sample=uniform_sample)
        dataset_list.append(dataset)
    return dataset_list