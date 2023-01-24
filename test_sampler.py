import torch
import os
import sys
import argparse
import importlib
import shutil
import numpy as np
import datetime
from time import time
import open3d as o3d
import logging
from pathlib import Path
from utils import distance_metric
from utils.modelnet_data_loading import ModelNet40_h5_dataset_list
from utils.sample_strategies import fps_sampling, random_sampling, index_points, fps_sampling_complete
from tqdm import tqdm
from utils.neighbor_search import query_knn_point
import math
from utils.large_modelnet_data_loading import ModelNetDataLoader
from utils.loss_functions import TaskLoss_cls
import copy
from utils.visualization import plot_3d_point_cloud, visu_pc_w_dis, savetxt_pc_w_dis


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='OR-PD-test')
    parser.add_argument('--task_model', type=str, default='pointnet_cls_vanilla', help='default=pointnet_cls_vanilla')
    parser.add_argument('--sample_model', type=str, default='ASPD_dgcnn')
    parser.add_argument('--coarse_log_dir', type=str, default=None, help='other pretrained model')
    parser.add_argument('--log_dir', type=str, default='sampler_cls', help='OR-PD_general_chamfer(16-512) / SNP_our(16-512)')
    parser.add_argument('--task_log_dir', type=str, default='baseline_aug', help='default=baseline_aug')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_point', type=int, default=1024)
    parser.add_argument('--n_sample', type=int, default=256)

    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--data_dir', type=str, default='modelnet40_ply_hdf5_2048')

    parser.add_argument('--random_seed', type=int, default=8, help='int or None')

    parser.add_argument('--pre_sample', type=str, default='fps')

    parser.add_argument('--use_atten', type=bool, default=True)

    parser.add_argument('--dense_eval', type=bool, default=False)

    parser.add_argument('--use_fps', type=bool, default=True)

    parser.add_argument('--restore_last', type=bool, default=True)

    parser.add_argument('--visu', type=bool, default=False)

    parser.add_argument('--class_choice', type=int, default=None)

    return parser.parse_args()
FLAGS = parse_args()

def pred2acc(pred, label):
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(label.long().data).cpu().sum()
    acc = correct.item() / float(label.size()[0])
    return acc

def creat_dir(args):
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    task_log_dir = exp_dir.joinpath(args.task_model)

    task_log_dir = task_log_dir.joinpath(args.task_log_dir)
    task_checkpoints_dir = task_log_dir.joinpath('checkpoints/')

    exp_dir = exp_dir.joinpath(args.sample_model)
    exp_dir.mkdir(exist_ok=True)

    if args.coarse_log_dir is not None:
        coarse_checkpoints_dir = exp_dir.joinpath(args.coarse_log_dir)
        coarse_checkpoints_dir = coarse_checkpoints_dir.joinpath('checkpoints/')
    else:
        coarse_checkpoints_dir = None

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    return log_dir, checkpoints_dir, task_checkpoints_dir, coarse_checkpoints_dir


def data_loading(args):
    '''data loading'''
    train_dataset_list = ModelNet40_h5_dataset_list(dataset_name=args.data_dir, split='train', n_point=args.n_point)
    test_dataset_list = ModelNet40_h5_dataset_list(dataset_name=args.data_dir, split='test', n_point=args.n_point, class_choice=args.class_choice)
    num_train_data = 0
    num_test_data = 0
    for loader_id in range(len(train_dataset_list)):
        num_train_data += len(train_dataset_list[loader_id])
    for loader_id in range(len(test_dataset_list)):
        num_test_data += len(test_dataset_list[loader_id])
    print('num of train_data: %d' % num_train_data)
    print('num of test_data: %d' % num_test_data)
    return train_dataset_list, test_dataset_list


def log(args, log_dir):
    '''log'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_checkpoints(model, checkpoints_dir, source_checkpoints_dir=None):
    try:
        if FLAGS.restore_last:
            checkpoint = torch.load(str(checkpoints_dir) + '/last_model.pth', map_location=torch.device(cur_device))
        else:
            # checkpoint = torch.load(str(checkpoints_dir) + '/epoch_50.pth', map_location=torch.device(cur_device))
            checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth', map_location=torch.device(cur_device))
        model_state_dict = checkpoint['model_state_dict']
        sampler_dict = {k: v for k, v in model_state_dict.items() if 'sampler' in k}
        model_state_dict = model.state_dict()
        model_state_dict.update(sampler_dict)
        model.load_state_dict(model_state_dict)
        print('Use pre-trained models')
    except:
        if FLAGS.restore_last:
            checkpoint = torch.load(str(checkpoints_dir) + '/last_model.pth', map_location=torch.device(cur_device))
        else:
            # checkpoint = torch.load(str(checkpoints_dir) + '/epoch_50.pth', map_location=torch.device(cur_device))
            checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth', map_location=torch.device(cur_device))
        model_state_dict = checkpoint['model_state_dict']
        sampler_dict = {k: v for k, v in model_state_dict.items() if 'sampler' in k}
        model_state_dict = model.state_dict()
        model_state_dict.update(sampler_dict)
        model.load_state_dict(model_state_dict)
        # print('Use pre-trained models')
    return model, None


def main(args):
    def log_string(string):
        logger.info(string)
        print(string)

    '''log'''
    log_dir, checkpoints_dir, task_checkpoints_dir, source_checkpoints_dir = creat_dir(args)
    logger = log(args, log_dir)
    log_string('PARAMETER ...')
    log_string(args)

    shutil.copy(str(os.path.abspath(__file__)), str(log_dir))

    '''data'''
    print('data loading')
    train_dataset_list, test_dataset_list = data_loading(args)


    '''create model'''
    MODEL = importlib.import_module(args.sample_model)
    sampler = MODEL.get_model(use_atten=args.use_atten, pre_sample=args.pre_sample).to(cur_device)

    TASK_MODEL = importlib.import_module(args.task_model)
    task_model = TASK_MODEL.get_model().to(cur_device)
    task_model.requires_grad_(False)
    task_model.eval()
    task_checkpoints = torch.load(str(task_checkpoints_dir) + '/best_model.pth', map_location=torch.device(cur_device))
    task_model.load_state_dict(task_checkpoints['model_state_dict'])

    task_model.sampler = sampler

    task_model, start_epoch = load_checkpoints(task_model, checkpoints_dir)

    # # compute memory cost
    # num_param = 0
    # for name, param in task_model.named_parameters():
    #     if 'sampler' in name:
    #         num_param += param.numel()
    # print(num_param)

    if args.dense_eval:
        # n_sample_list = np.array([16,32,64,128,256,512])
        n_sample_list = 2 ** np.linspace(4, 10, 50)
        n_sample_list = n_sample_list.astype('int')
    else:
        n_sample_list = [args.n_sample]


    with torch.no_grad():
        task_model.sampler.eval()
        Acc_list = []
        RSR_list = []
        CD_list = []
        HD_list = []
        for n_sample in n_sample_list:
            mean_correct = []
            mean_cd = []
            mean_hd = []
            repeat_sample_cnt = []
            for dataset_id in range(len(test_dataset_list)):
                print('test_file %d' % dataset_id)
                data_loader = torch.utils.data.DataLoader(test_dataset_list[dataset_id], batch_size=args.batch_size, shuffle=False)
                for batch_id, (points, label) in tqdm(enumerate(data_loader), total=len(data_loader)):

                    k_neighbors = int(40 * (args.n_point / 1024))
                    # points = points[:,:n_input,:]
                    # k_neighbors=40
                    points, label = points.to(cur_device).float(), label.to(cur_device).long()

                    if args.use_fps:
                        if args.pre_sample=='fps':
                            pred_subset, _ = fps_sampling(points, n_sample, seed=args.random_seed)
                        else:
                            pred_subset, _ = random_sampling(points, n_sample, seed=args.random_seed)
                    else:
                        pred_subset, _ = task_model.sampler(points.transpose(2, 1), n_sample, k_neighbors)

                    simplified_subset = pred_subset

                    cd = distance_metric.chamfer_dist(pred_subset, points)
                    hd = distance_metric.hausdorff_dist(pred_subset, points)
                    mean_cd.append(cd.item())
                    mean_hd.append(hd.item())
                    pred = task_model(pred_subset.transpose(2,1))
                    acc = pred2acc(pred,label)
                    if args.visu:
                        '''used to visualize sampled results before and after matching'''
                        simplified_subset_np = simplified_subset.cpu().data.numpy()
                        pred_subset_np = pred_subset.cpu().data.numpy()
                        points_np = points.cpu().data.numpy()

                        for b in range(pred_subset_np.shape[0]):
                            visu_pc_w_dis(points_np[b], 'input points')
                            visu_pc_w_dis(pred_subset_np[b], 'sampled points')

                            savetxt_pc_w_dis(points_np[b],'FPS_input_airplane'+str(b),'FPS_'+str(args.n_sample))
                            savetxt_pc_w_dis(pred_subset_np[b], 'FPS_sample_airplane'+str(b),'FPS_'+str(args.n_sample))
                        exit()

                    mean_correct.append(acc)

            test_mean_acc = np.mean(mean_correct)
            test_mean_cd = np.mean(mean_cd)
            test_mean_hd = np.mean(mean_hd)
            repeat_sample_rate = np.mean(repeat_sample_cnt) / n_sample
            log_string('test acc (%d points) %.5f, cd %.5f, hd %.5f' % (n_sample, test_mean_acc, test_mean_cd, test_mean_hd))
            Acc_list.append(100 * test_mean_acc)
            CD_list.append(100 * test_mean_cd)
            HD_list.append(100 * test_mean_hd)

            return Acc_list[-1]

        # return test_mean_acc
        result = np.concatenate([n_sample_list.reshape(-1, 1), np.array(Acc_list).reshape(-1, 1),
                                 np.array(CD_list).reshape(-1, 1), np.array(HD_list).reshape(-1, 1)], 1)
        # np.save('result/cls.npy', result)


if __name__ == '__main__':
    args = parse_args()
    main(args)

