import torch
import os
import sys
import argparse
import importlib
import shutil
import numpy as np
import datetime
import logging
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='OR-PD-train')
    parser.add_argument('--task_model', type=str, default='pcrnet')
    parser.add_argument('--sample_model', type=str, default='SampleNet')
    parser.add_argument('--coarse_log_dir', type=str, default=None, help='other pretrained model')
    parser.add_argument('--log_dir', type=str, default='SNP_regis')
    parser.add_argument('--task_log_dir', type=str, default='pcrnet', help='full_acc=87.3%')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', default=300)
    parser.add_argument('--start_epoch', default=None)
    parser.add_argument('--n_point', type=int, default=1024)
    parser.add_argument('--n_sample', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', default=0.001, help='initial learning rate； default=0.001/0.0001')
    parser.add_argument('--step_size', default=20, help='decay step for lr decay, default:20/10')
    parser.add_argument('--lr_decay', default=0.7, help='decay rate for lr decay,default:0.2')

    parser.add_argument('--exp_decay', type=bool, default=False, help='exponentially decay, step by step')

    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--data_dir', type=str, default='modelnet40_ply_hdf5_2048')

    parser.add_argument('--random_seed', type=int, default=0, help='int or None')

    parser.add_argument('--trainable_tmp', type=bool, default=False)

    parser.add_argument('--pre_sample', type=str, default='fps')

    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--w_sigma', type=float, default=0.01, help='weights for sigma')

    parser.add_argument('--S_beta', type=float, default=1, help='beta in Simplify loss')
    parser.add_argument('--S_gamma', type=float, default=1)
    parser.add_argument('--S_delta', type=float, default=0)

    parser.add_argument('--lmbda', type=float, default=100, help='weight for task loss')

    parser.add_argument('--min_sigma', type=float, default=1e-2)
    parser.add_argument('--initial_temp', type=float, default=1)

    parser.add_argument('--partial_train', type=bool, default=False)

    parser.add_argument('--save_last', type=bool, default=True)

    parser.add_argument('--uni_train', type=bool, default=False)

    parser.add_argument('--use_uniform_sample', type=bool, default=False)

    parser.add_argument('--use_large_dataset', type=bool, default=False)

    parser.add_argument('--process_data', type=bool, default=True)

    parser.add_argument('--check_repeat', type=bool, default=False)

    parser.add_argument('--use_atten', type=bool, default=False)

    parser.add_argument('--use_match', type=bool, default=True)

    parser.add_argument('--skip_projection', type=bool, default=False)

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


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def optimizer_loading(args, model):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            # classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    return optimizer


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def momentum_lr_adjusting(args, model, optimizer, epoch):
    '''adjust learning rate and BN momentum'''
    LEARNING_RATE_CLIP = 1e-5  # default=1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
    print('Learning rate:%f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
    if momentum < 0.01:
        momentum = 0.01
    print('BN momentum updated to: %f' % momentum)
    model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
    return model, optimizer


def check_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


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

    if args.exp_decay:
        min_lr = 1e-5
        init_lr = args.learning_rate
        lr_decay = math.exp(math.log(min_lr/init_lr)/args.epoch)
        args.lr_decay = lr_decay
        args.step_size = 1

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
        sampler = MODEL.get_model(num_out_points=args.n_point, all_epoch=args.epoch,
                                  initial_temperature=args.initial_temp, trainable_tmp=args.trainable_tmp,
                                  min_sigma=args.min_sigma).to(cur_device)
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

    '''optimizer'''
    if args.partial_train:
        for name, param in task_model.sampler.named_parameters():
            if 'convs' in name or 'bns' in name or 'sa1' in name or 'SELayer' in name:
                param.requires_grad=True
            else:
                param.requires_grad=False

    check_trainable_parameters(task_model)

    optimizer = optimizer_loading(args, task_model)

    try:
        checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth', map_location=torch.device(cur_device))
        optimizer.load_state_dict(checkpoint["optimizer"])
    except:
        pass


    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    best_re = 1e10
    n_sample_list = np.array([16,32,64,128,256,512])


    for epoch in range(start_epoch, args.epoch):
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=args.workers)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.workers)
        task_model, optimizer = momentum_lr_adjusting(args, task_model, optimizer, epoch)
        task_model.sampler.train()
        # check_trainable_parameters(task_model)

        mean_loss = []
        train_mean_correct = []
        sigma_list = []

        for batch_id, (data) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            # points = points.data.numpy()
            # # points[:, :, 0:3] = provider.rotate_point_cloud(points[:, :, 0:3])
            # # points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
            # points = provider.random_point_dropout(points, max_dropout_ratio=0.9)
            # points = torch.Tensor(points)

            template, source, igt = data
            template, source = template.to(cur_device).float(), source.to(cur_device).float()

            if args.uni_train:
                # n_sample = np.random.choice(n_sample_list)
                n_sample = n_sample_list[batch_id % len(n_sample_list)]
            else:
                n_sample = args.n_sample

            # check_trainable_parameters(task_model)

            if args.skip_projection:
                pred_subset_t, pre_sampled_subset_t = task_model.sampler(template.transpose(2,1), n_sample)
                pred_subset_s, pre_sampled_subset_s = task_model.sampler(source.transpose(2,1), n_sample)
                samp_data = (pred_subset_t, pred_subset_s, igt)
            else:
                proj_subset_t, pred_subset_t, pre_sampled_subset_t = task_model.sampler(template.transpose(2, 1), n_sample, epoch=epoch)
                proj_subset_s, pred_subset_s, pre_sampled_subset_s = task_model.sampler(source.transpose(2, 1), n_sample, epoch=epoch)
                samp_data = (proj_subset_t, proj_subset_s, igt)

            if args.sample_model=='SNET' or args.sample_model=='SampleNet':
                n_sample = 16
                while n_sample<=1024:
                    subset_pred_t = pred_subset_t[:, :n_sample, :]
                    subset_pred_s = pred_subset_s[:, :n_sample, :]
                    subset_proj_t = proj_subset_t[:,:n_sample,:]
                    subset_proj_s = proj_subset_s[:,:n_sample,:]
                    samp_data = (subset_proj_t, subset_proj_s, igt)
                    task_loss, pcrnet_loss_info = task_criterion(samp_data, task_model)
                    sample_loss_t = sample_criterion(template, subset_pred_t)
                    sample_loss_s = sample_criterion(source, subset_pred_s)
                    sample_loss = 0.5 * (sample_loss_t + sample_loss_s)

                    if n_sample==16:
                        total_loss = args.lmbda*task_loss + sample_loss
                    else:
                        total_loss = total_loss + args.lmbda*task_loss + sample_loss
                    n_sample *= 2
            else:
                offset_t = pred_subset_t - pre_sampled_subset_t
                offset_s = pred_subset_s - pre_sampled_subset_s
                sample_loss_t = sample_criterion(template, pred_subset_t, offset=offset_t)
                sample_loss_s = sample_criterion(source, pred_subset_s, offset=offset_s)
                sample_loss = sample_loss_t + sample_loss_s

                task_loss, pcrnet_loss_info = task_criterion(samp_data, task_model)
                total_loss = args.lmbda*task_loss + sample_loss

            if args.trainable_tmp:
                sigma = task_model.sampler.project.sigma(epoch=epoch)
                total_loss = total_loss + sigma * args.w_sigma
                sigma_list.append(sigma.item())

            rot_err = pcrnet_loss_info['rot_err']
            train_mean_correct.append(rot_err)

            total_loss.backward()
            mean_loss.append(total_loss.item())
            optimizer.step()

        log_string('epoch %d, train loss %.5f, sigma %.5f' % (epoch, np.mean(mean_loss), np.mean(sigma_list)))
        # log_string('epoch %d, train acc %.5f' % (epoch, np.mean(train_mean_correct)))

        # if epoch%10!=0 and epoch!=(args.epoch-1) and args.uni_train:
        #     continue

        with torch.no_grad():
            task_model.sampler.eval()
            mean_RE = []
            repeat_sample_cnt = []
            for batch_id, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                template, source, igt = data
                template, source = template.to(cur_device).float(), source.to(cur_device).float()

                # if args.uni_train:
                #     # n_sample = np.random.choice(n_sample_list)
                #     n_sample = n_sample_list[batch_id%len(n_sample_list)]
                # else:
                #     n_sample = args.n_sample
                n_sample = args.n_sample

                pred_subset_t, _ = task_model.sampler(template.transpose(2,1), n_sample)
                pred_subset_s, _ = task_model.sampler(source.transpose(2,1), n_sample)

                if args.sample_model == 'SNET' or args.sample_model=='SampleNet':
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

                samp_data = (pred_subset_t, pred_subset_s, igt)



                _, pcrnet_loss_info = task_criterion(samp_data, task_model)
                rot_err = pcrnet_loss_info['rot_err']

                mean_RE.append(rot_err.cpu().data.numpy())

            test_mean_re = np.mean(mean_RE)
            repeat_sample_rate = np.mean(repeat_sample_cnt) / args.n_sample
            log_string('epoch %d, test rot err (%d points) %.5f, repeat sample rate %.5f' % (epoch, args.n_sample, test_mean_re, repeat_sample_rate))


        if test_mean_re < best_re:
            best_re = test_mean_re
            log_string('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': task_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if epoch==(args.epoch-1) and args.save_last:
            log_string('Save model...')
            savepath = str(checkpoints_dir) + '/last_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': task_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')


    return best_re


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