import torch
import os
import sys
import argparse
import importlib
import shutil
import numpy as np
import datetime
import logging
from utils import distance_metric
from pathlib import Path
from utils.modelnet_data_loading import ModelNet40_h5_dataset_list
from utils.sample_strategies import fps_sampling, random_sampling, index_points
from tqdm import tqdm
from utils.neighbor_search import query_knn_point
import math
from utils.large_modelnet_data_loading import ModelNetDataLoader
from utils.loss_functions import TaskLoss_regis
from utils.regis_utils.qdataset import QuaternionFixedDataset
from data.regis_modelnet_data_loading import ModelNetCls
import torchvision
from utils.regis_utils.pctransforms import OnUnitCube, PointcloudToTensor
from torch.utils.data import DataLoader
from utils.visualization import visu_pc_w_dis, savetxt_pc_w_dis


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='OR-PD-train')
    parser.add_argument('--task_model', type=str, default='pcrnet')
    parser.add_argument('--sample_model', type=str, default='ASPD_dgcnn')
    parser.add_argument('--progressive_version', type=bool, default=True)
    parser.add_argument('--coarse_log_dir', type=str, default=None, help='other pretrained model')
    parser.add_argument('--log_dir', type=str, default='sampler_regis')
    parser.add_argument('--task_log_dir', type=str, default='pcrnet', help='full_acc=87.3%')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--start_epoch', default=None)
    parser.add_argument('--n_point', type=int, default=1024)
    parser.add_argument('--n_sample', type=int, default=16)
    parser.add_argument('--visu', type=bool, default=True)

    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--data_dir', type=str, default='modelnet40_ply_hdf5_2048')

    parser.add_argument('--random_seed', type=int, default=0, help='int or None')

    parser.add_argument('--use_fps', type=bool, default=True)

    parser.add_argument('--pre_sample', type=str, default='rs')

    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--w_sigma', type=float, default=0.1, help='weights for sigma')

    parser.add_argument('--S_beta', type=float, default=0, help='beta in Simplify loss')
    parser.add_argument('--S_gamma', type=float, default=1)
    parser.add_argument('--S_delta', type=float, default=0)

    parser.add_argument('--lmbda', type=float, default=100, help='weight for task loss')

    parser.add_argument('--min_sigma', type=float, default=1e-4)
    parser.add_argument('--initial_temp', type=float, default=0.1)

    parser.add_argument('--partial_train', type=bool, default=False)

    parser.add_argument('--restore_last', type=bool, default=False)

    parser.add_argument('--uni_train', type=bool, default=False)

    parser.add_argument('--use_uniform_sample', type=bool, default=False)

    parser.add_argument('--use_large_dataset', type=bool, default=False)

    parser.add_argument('--process_data', type=bool, default=True)

    parser.add_argument('--check_repeat', type=bool, default=True)

    parser.add_argument('--use_atten', type=bool, default=True)

    parser.add_argument('--use_match', type=bool, default=True)

    parser.add_argument('--skip_projection', type=bool, default=True)

    parser.add_argument('--specific_mode', type=bool, default=False, help='extract knn feature for attention module from knn neighbors')

    parser.add_argument('--dense_eval', type=bool, default=False)


    ##### register related #####
    parser.add_argument('--exp_name', type=str, default='log/pcr_net/car_ipcrnet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset')  # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--pretrained', default='log/pcr_net/car_ipcrnet/checkpoints/best_model.t7', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    parser.add_argument('--seed', type=int, default=4321)

    parser.add_argument('--datafolder', type=str, default='car_hdf5_2048')

    return parser.parse_args()
FLAGS = parse_args()


def creat_dir(args):
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)

    task_log_dir = Path(args.exp_name)
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
    test_dataset_list = ModelNet40_h5_dataset_list(dataset_name=args.data_dir, split='test', n_point=args.n_point)
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
    for b in range(idx.shape[0]):
        elements, counts = torch.unique(idx[b], return_counts=True)
        cnt += sum(counts != 1)

    return cnt.cpu().data.numpy()


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
        if args.restore_last:
            checkpoint = torch.load(str(checkpoints_dir) + '/last_model.pth', map_location=torch.device(cur_device))
        else:
            checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth', map_location=torch.device(cur_device))
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Use pre-trained models')
    except:
        if source_checkpoints_dir is not None:
            checkpoint = torch.load(str(source_checkpoints_dir) + '/best_model.pth',
                                    map_location=torch.device(cur_device))
            model_state_dict = checkpoint['model_state_dict']
            cur_state_dict = model.state_dict()
            cur_state_dict.update(model_state_dict)
            model.load_state_dict(cur_state_dict)
            # model.load_state_dict(checkpoint['model_state_dict'])
            print('No existing models, starting training from source checkpoints...')
        else:
            print('No existing models, starting training from scratch...')
            # model = model.apply(weights_init)
        start_epoch = 0
    return model, start_epoch


def main(args):
    def log_string(string):
        logger.info(string)
        print(string)

    '''log'''
    log_dir, checkpoints_dir, task_checkpoints_dir, source_checkpoints_dir = creat_dir(args)
    logger = log(args, log_dir)
    log_string('PARAMETER ...')
    log_string(args)

    shutil.copy('models/%s.py' % args.sample_model, str(log_dir))
    shutil.copy(str(os.path.abspath(__file__)), str(log_dir))

    '''data'''
    # '''data'''
    transforms = torchvision.transforms.Compose([PointcloudToTensor(), OnUnitCube()])

    traindata = ModelNetCls(
        args.num_points,
        transforms=transforms,
        train=True,
        download=False,
        folder=args.datafolder,
    )
    testdata = ModelNetCls(
        args.num_points,
        transforms=transforms,
        train=False,
        download=False,
        folder=args.datafolder,
    )

    train_repeats = max(int(5000 / len(traindata)), 1)

    trainset = QuaternionFixedDataset(traindata, repeat=train_repeats, seed=0, )
    testset = QuaternionFixedDataset(testdata, repeat=1, seed=0)

    '''create model'''
    MODEL = importlib.import_module(args.sample_model)
    if args.sample_model=='SNET' or args.sample_model=='SampleNet':
        sampler = MODEL.get_model(num_out_points=args.n_point).to(cur_device)
    else:
        sampler = MODEL.get_model(use_atten=args.use_atten, pre_sample=args.pre_sample).to(cur_device)

    TASK_MODEL = importlib.import_module(args.task_model)
    task_model = TASK_MODEL.iPCRNet().to(cur_device)
    task_model.requires_grad_(False)
    task_model.eval()
    task_checkpoints = torch.load(str(task_checkpoints_dir) + '/best_model.t7', map_location=torch.device(cur_device))
    task_model.load_state_dict(task_checkpoints)

    task_model.sampler = sampler

    task_model, start_epoch = load_checkpoints(task_model, checkpoints_dir, source_checkpoints_dir)

    sample_criterion = MODEL.get_sample_loss(alpha=args.alpha, beta=args.beta, S_beta=args.S_beta, S_gamma=args.S_gamma, S_delta=args.S_delta).to(cur_device)
    task_criterion = TaskLoss_regis().to(cur_device)

    if args.dense_eval:
        # n_sample_list = np.array([16,32,64,128,256,512])
        n_sample_list = 2 ** np.linspace(4, 10, 50)
        n_sample_list = n_sample_list.astype('int')
    else:
        n_sample_list = [args.n_sample]

    with torch.no_grad():
        task_model.sampler.eval()
        NRE_list = []
        RSR_list = []
        HD_list = []
        for n_sample in n_sample_list:
            mean_RE = []
            mean_HD = []
            repeat_sample_cnt = []
            test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     num_workers=args.workers)
            for batch_id, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                template, source, igt = data
                template, source = template.to(cur_device).float(), source.to(cur_device).float()

                if args.use_fps:
                    if args.pre_sample == 'fps':
                        pred_subset_t = fps_sampling(template, n_sample)[0]
                        pred_subset_s = fps_sampling(source, n_sample)[0]
                    else:
                        pred_subset_t = random_sampling(template, n_sample)[0]
                        pred_subset_s = random_sampling(source, n_sample)[0]
                else:
                    pred_subset_t, _ = task_model.sampler(template.transpose(2,1), n_sample)
                    pred_subset_s, _ = task_model.sampler(source.transpose(2,1), n_sample)

                    if args.sample_model == 'SNET' or args.sample_model == 'SampleNet':
                        if args.progressive_version:
                            pred_subset_t = pred_subset_t[:, :n_sample, :]
                            pred_subset_s = pred_subset_s[:, :n_sample, :]


                    if args.use_match:
                        idx_t = query_knn_point(nsample=1, xyz=template, new_xyz=pred_subset_t)
                        idx_s = query_knn_point(nsample=1, xyz=source, new_xyz=pred_subset_s)
                        pred_subset_t = index_points(template, idx_t.squeeze())
                        pred_subset_s = index_points(source, idx_s.squeeze())

                        if args.check_repeat:
                            cnt_t = sample_idx_check(idx_t)
                            cnt_s = sample_idx_check(idx_s)
                            cnt = 0.5 * (cnt_t + cnt_s)
                        else:
                            cnt = 0
                        repeat_sample_cnt.append(cnt/template.shape[0])

                hd_t = distance_metric.hausdorff_dist(pred_subset_t, template)
                hd_s = distance_metric.hausdorff_dist(pred_subset_s, source)
                hd = 0.5 * (hd_t + hd_s)
                mean_HD.append(hd.item())

                samp_data = (pred_subset_t, pred_subset_s, igt)

                # if args.sample_model == 'SNET':
                #     if args.progressive_version:
                #         subset_t = pred_subset_t[:, :n_sample, :]
                #         subset_s = pred_subset_s[:, :n_sample, :]
                #         samp_data = (subset_t, subset_s, igt)

                _, pcrnet_loss_info = task_criterion(samp_data, task_model)
                rot_err = pcrnet_loss_info['rot_err']

                mean_RE.append(rot_err.cpu().data.numpy())

                if args.visu and batch_id in [2,21]:
                    print(f'batch {batch_id}, rot err: {rot_err}')
                    est_transform = pcrnet_loss_info['est_transform']
                    transform_source = est_transform.rotate(template)
                    transform_source_np = transform_source.cpu().data.numpy()
                    template_np = template.cpu().data.numpy()
                    source_np = source.cpu().data.numpy()
                    pred_subset_t_np = pred_subset_t.cpu().data.numpy()
                    pred_subset_s_np = pred_subset_s.cpu().data.numpy()
                    for b in range(pred_subset_t.shape[0]):
                        template_np_b = template_np[b]
                        source_np_b = source_np[b]
                        transform_source_np_b = transform_source_np[b]
                        pred_subset_s_np_b = pred_subset_s_np[b]
                        pred_subset_t_np_b = pred_subset_t_np[b]
                        sampler_name = 'RS'
                        dir_name = sampler_name + '_regis_' + str(args.n_sample)
                        savetxt_pc_w_dis(template_np_b, dirname=dir_name, filename=sampler_name+'_batch'+str(batch_id)+'ins'+str(b)+'_template')
                        savetxt_pc_w_dis(source_np_b, dirname=dir_name, filename=sampler_name+'_batch'+str(batch_id)+'ins'+str(b)+'_source')
                        savetxt_pc_w_dis(transform_source_np_b, dirname=dir_name, filename=sampler_name+'_batch'+str(batch_id)+'ins'+str(b)+'_tran_source')
                        savetxt_pc_w_dis(pred_subset_t_np_b, dirname=dir_name, filename=sampler_name+'_batch'+str(batch_id)+'ins'+str(b)+'_subset_t')
                        savetxt_pc_w_dis(pred_subset_s_np_b, dirname=dir_name, filename=sampler_name+'_batch'+str(batch_id)+'ins'+str(b)+'_subset_s')




            test_mean_re = np.mean(mean_RE)
            test_mean_hd = np.mean(mean_HD)
            repeat_sample_rate = np.mean(repeat_sample_cnt) / n_sample
            log_string('test rot err (%d points) %.5f, repeat sample rate %.5f' % (n_sample, test_mean_re, repeat_sample_rate))
            NRE_list.append(test_mean_re)
            RSR_list.append(repeat_sample_rate)
            HD_list.append(test_mean_hd)

        result = np.concatenate([n_sample_list.reshape(-1,1), np.array(NRE_list).reshape(-1,1),
                                 np.array(HD_list).reshape(-1,1)], 1)
        # np.save('result/new/regis_RS.npy', result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    exit()

    n_sample_list = [16, 64, 256]
    log_dir_list = ['base_16', 'base_64', 'base_256']


    for i in range(len(n_sample_list)):
        args.n_sample = n_sample_list[i]
        args.log_dir = log_dir_list[i]
        main(args)