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
    parser.add_argument('--cascade_way', type=bool, default=False)
    parser.add_argument('--progressive_version', type=bool, default=True, help='to check SNET whether work in progressive mode')
    parser.add_argument('--task_log_dir', type=str, default='baseline_aug', help='default=baseline_aug')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_point', type=int, default=1024)
    parser.add_argument('--n_sample', type=int, default=32)
    parser.add_argument('--actual_n_sample', type=int, default=32, help='just for SampleNet')

    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--data_dir', type=str, default='modelnet40_ply_hdf5_2048')

    parser.add_argument('--random_seed', type=int, default=8, help='int or None')

    parser.add_argument('--pre_sample', type=str, default='fps')

    parser.add_argument('--use_uniform_sample', type=bool, default=False)

    parser.add_argument('--use_large_dataset', type=bool, default=False)

    parser.add_argument('--process_data', type=bool, default=True)

    parser.add_argument('--check_repeat', type=bool, default=True)

    parser.add_argument('--use_atten', type=bool, default=True)

    parser.add_argument('--use_match', type=bool, default=False)

    parser.add_argument('--skip_projection', type=bool, default=True)

    parser.add_argument('--dense_eval', type=bool, default=False)

    parser.add_argument('--use_fps', type=bool, default=False)

    parser.add_argument('--restore_last', type=bool, default=True)

    parser.add_argument('--visu', type=bool, default=False)

    parser.add_argument('--visu_correction_only', type=bool, default=False)

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


def sample_idx_check(idx):
    cnt = 0
    repeat_elements_list = []
    for b in range(idx.shape[0]):
        elements, counts = torch.unique(idx[b], return_counts=True)
        cnt += sum(counts != 1)
        repeat_elements = elements[counts!=1].cpu().data.numpy()
        repeat_elements_list.append(repeat_elements)

    return cnt.cpu().data.numpy(), repeat_elements_list


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
    if args.use_large_dataset:
        # large scale modelnet40
        train_dataset_list = [ModelNetDataLoader(args, split='train', process_data=args.process_data)]
        test_dataset_list = [ModelNetDataLoader(args, split='test', process_data=args.process_data)]
    else:
        train_dataset_list, test_dataset_list = data_loading(args)
        # test_dataset_list, train_dataset_list = data_loading(args)


    '''create model'''
    MODEL = importlib.import_module(args.sample_model)
    if args.sample_model=='SNET':
        if args.progressive_version:
            sampler = MODEL.get_model(num_out_points=args.n_point).to(cur_device)
        else:
            sampler = MODEL.get_model(num_out_points=args.n_sample).to(cur_device)
    else:
        sampler = MODEL.get_model(use_atten=args.use_atten, pre_sample=args.pre_sample).to(cur_device)

    TASK_MODEL = importlib.import_module(args.task_model)
    task_model = TASK_MODEL.get_model().to(cur_device)
    task_model.requires_grad_(False)
    task_model.eval()
    task_checkpoints = torch.load(str(task_checkpoints_dir) + '/best_model.pth', map_location=torch.device(cur_device))
    task_model.load_state_dict(task_checkpoints['model_state_dict'])

    task_model.sampler = sampler

    if args.cascade_way:
        pre_task_model, start_epoch = load_checkpoints(task_model, source_checkpoints_dir)
        pre_sampler = copy.deepcopy(pre_task_model.sampler)
        for params in pre_sampler.parameters():
            params.requires_grad = False
        pre_sampler.eval()
        pre_sampler.cascade = False

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

                    # n_input = 1200
                    # k_neighbors = int(40 * (n_input / 1024))
                    # points = points[:,:n_input,:]
                    k_neighbors=40
                    points, label = points.to(cur_device).float(), label.to(cur_device).long()

                    if args.use_fps:
                        if args.pre_sample=='fps':
                            pred_subset, _ = fps_sampling(points, n_sample, seed=args.random_seed)
                        else:
                            pred_subset, _ = random_sampling(points, n_sample, seed=args.random_seed)
                    else:
                        if args.cascade_way:
                            source_subset = pre_sampler(points.transpose(2, 1), n_sample)
                            sample_idx = query_knn_point(nsample=1, xyz=points, new_xyz=source_subset).squeeze()
                            pred_subset = \
                                task_model.sampler(points.transpose(2, 1), n_sample,
                                                   source_subset=source_subset, sample_idx=sample_idx)
                        else:
                            # t1 = time()
                            pred_subset, _ = task_model.sampler(points.transpose(2, 1), n_sample, k_neighbors)
                    # print(task_model.sampler.project.sigma(epoch=0))
                    if args.sample_model=='SNET' or args.sample_model=='SampleNet':
                        if args.progressive_version:
                            pred_subset = pred_subset[:,:n_sample,:]

                    # t2 = time()
                    # print('all cost time(ms): ', 1000 * (t2 - t1) / points.shape[0])
                    simplified_subset = pred_subset

                    if args.use_match:
                        # pred_before_match = visu_pc_w_dis(pred_subset.cpu().data.numpy()[0], return_pcd=True)
                        idx = query_knn_point(nsample=1, xyz=points, new_xyz=pred_subset)
                        pred_subset = index_points(points, idx.squeeze())
                        # points_pcd = visu_pc_w_dis(points.cpu().data.numpy()[0], return_pcd=True)
                        # pred_after_match = visu_pc_w_dis(pred_subset.cpu().data.numpy()[0], return_pcd=True)
                        #
                        # pred_before_match.paint_uniform_color([0,0,1])
                        # points_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                        # pred_after_match.paint_uniform_color([1,0,0])
                        #
                        # o3d.visualization.draw_geometries([pred_before_match, points_pcd, pred_after_match])
                        # print(label)
                        if args.check_repeat:
                            cnt, repeat_idx_list = sample_idx_check(idx)
                        else:
                            cnt = 0
                        repeat_sample_cnt.append(cnt/points.shape[0])

                        if args.sample_model=='SNET' and not args.progressive_version:
                            if args.n_sample != args.actual_n_sample:
                                pred_subset = fps_sampling_complete(points, args.actual_n_sample, idx.squeeze())[0]

                    # t2 = time()
                    # print('cost time(ms): ', 1000*(t2-t1)/points.shape[0])
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

                        # samp_label_npy = np.zeros((points_np.shape[0], points_np.shape[1]))
                        # samp_idx_npy = idx.cpu().data.numpy().squeeze()
                        # batch_idx = np.arange(points.shape[0]).reshape(-1, 1).repeat(samp_idx_npy.shape[1], 1)
                        # samp_label_npy[batch_idx, samp_idx_npy] = 1
                        # # for b in range(samp_idx_npy.shape[0]):
                        # #     samp_label_npy[b, repeat_idx_list[b]] = 1
                        # pc_after_match = np.concatenate([points_np, samp_label_npy.reshape(points_np.shape[0],-1,1)], axis=-1)
                        #
                        # pc_before_match = np.concatenate([points_np, simplified_subset_np],axis=1)
                        # simplified_label_npy = np.zeros((pc_before_match.shape[0], pc_before_match.shape[1]))
                        # simplified_label_npy[:,points_np.shape[1]:] = 1
                        # pc_before_match = np.concatenate([pc_before_match, simplified_label_npy.reshape(simplified_label_npy.shape[0],-1,1)], axis=-1)

                        for b in range(pred_subset_np.shape[0]):
                            # pc_before_match_b = pc_before_match[b]
                            # pc_after_match_b = pc_after_match[b]

                            # savetxt_pc_w_dis(pc_before_match_b,'SNP_before_match_airplane0','match_256')
                            # savetxt_pc_w_dis(pc_after_match_b, 'SNET_after_match_airplane0','match_256')

                            visu_pc_w_dis(points_np[b], 'input points')
                            visu_pc_w_dis(pred_subset_np[b], 'sampled points')

                            savetxt_pc_w_dis(points_np[b],'FPS_input_airplane'+str(b),'FPS_'+str(args.n_sample))
                            savetxt_pc_w_dis(pred_subset_np[b], 'FPS_sample_airplane'+str(b),'FPS_'+str(args.n_sample))
                        exit()

                        '''used to visualize different sampled results'''
                        # fps_subset, fps_idx = fps_sampling(points, n_sample, seed=args.random_seed)
                        # pc_npy = points.cpu().data.numpy()
                        # samp_label_npy = np.zeros((pc_npy.shape[0], pc_npy.shape[1]))
                        # fps_samp_label_npy = np.zeros((pc_npy.shape[0], pc_npy.shape[1]))
                        # samp_idx_npy = idx.cpu().data.numpy().squeeze()
                        # fps_samp_idx_npy = fps_idx.cpu().data.numpy().squeeze()
                        # batch_idx = np.arange(points.shape[0]).reshape(-1, 1).repeat(samp_idx_npy.shape[1], 1)
                        # samp_label_npy[batch_idx, samp_idx_npy] = 1
                        # fps_samp_label_npy[batch_idx, fps_samp_idx_npy] = 1
                        # pc_label_npy = np.concatenate([pc_npy, samp_label_npy.reshape(pc_npy.shape[0], -1, 1),
                        #                                fps_samp_label_npy.reshape(pc_npy.shape[0], -1, 1)], -1)
                        #
                        # fps_pred = task_model(fps_subset.transpose(2, 1))
                        # for b in range(pred.shape[0]):
                        #     pred_b = pred[b].reshape(1,-1)
                        #     fps_pred_b = fps_pred[b].reshape(1,-1)
                        #     label_b = label[b].reshape(1,-1)
                        #     fps_acc_b = pred2acc(fps_pred_b, label_b)
                        #     acc_b = pred2acc(pred_b, label_b)
                        #     if acc_b>fps_acc_b or not args.visu_correction_only:
                        #         visu_pc_w_dis(np.delete(pc_label_npy[b, :, :], -2, 1), str(b) + 'fps')
                        #         visu_pc_w_dis(np.delete(pc_label_npy[b, :, :], -1, 1), str(b) + 'learn')

                    mean_correct.append(acc)

            test_mean_acc = np.mean(mean_correct)
            test_mean_cd = np.mean(mean_cd)
            test_mean_hd = np.mean(mean_hd)
            repeat_sample_rate = np.mean(repeat_sample_cnt) / n_sample
            log_string('test acc (%d points) %.5f, cd %.5f, hd %.5f' % (n_sample, test_mean_acc, test_mean_cd, test_mean_hd))
            Acc_list.append(100 * test_mean_acc)
            CD_list.append(100 * test_mean_cd)
            HD_list.append(100 * test_mean_hd)

        # return test_mean_acc
        result = np.concatenate([n_sample_list.reshape(-1, 1), np.array(Acc_list).reshape(-1, 1),
                                 np.array(CD_list).reshape(-1, 1), np.array(HD_list).reshape(-1, 1)], 1)
        # np.save('result/acorss_sizes/cls_64.npy', result)



if __name__ == '__main__':
    args = parse_args()
    main(args)
    exit()

    # n_sample_list = [32, 64, 128, 256, 512, 1024]
    # log_dir_list = ['OR-PD_32_chamfer', 'OR-PD_64_chamfer', 'OR-PD_128_chamfer', 'OR-PD_256_chamfer', 'OR-PD_512_chamfer']
    #
    # Acc_npy = np.zeros((6,6))
    # for i in range(len(log_dir_list)):
    #     args.log_dir = log_dir_list[i]
    #     cur_acc_list = []
    #     for n_sample in n_sample_list:
    #         args.n_sample = n_sample
    #         acc = main(args)
    #         cur_acc_list.append(acc)
    #     Acc_npy[i] = np.array(cur_acc_list)

    # args.use_fps = True
    # cur_acc_list = []
    # for n_sample in n_sample_list:
    #     args.n_sample = n_sample
    #     acc = main(args)
    #     cur_acc_list.append(acc)
    # Acc_npy[-1] = np.array(cur_acc_list)
    #
    # np.save('result/Exp-explanation-robust.npy', Acc_npy)

    cur_acc_list = []
    n_sample_list = 2 ** np.linspace(4, 10, 2)
    n_sample_list = n_sample_list.astype('int')
    for n_sample in n_sample_list:
        args.n_sample = n_sample
        acc = main(args)
        cur_acc_list.append(acc)
    acc_np = np.array(cur_acc_list) * 100
    n_sample_acc = np.concatenate([n_sample_list.reshape(-1,1), acc_np.reshape(-1,1)], -1)

    np.save('result/new/cls_standard.npy', n_sample_acc)

