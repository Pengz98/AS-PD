import torch
import os
import sys
import pynvml
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
from utils.loss_functions import TaskLoss_cls
import copy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='OR-PD-train')
    parser.add_argument('--task_model', type=str, default='pointnet_cls_vanilla')
    parser.add_argument('--sample_model', type=str, default='ASPD_dgcnn')
    parser.add_argument('--coarse_log_dir', type=str, default=None, help='other pretrained model')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--task_log_dir', type=str, default='baseline_aug', help='full_acc=87.3%')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', default=200)
    parser.add_argument('--start_epoch', default=None)
    parser.add_argument('--n_point', type=int, default=1024)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', default=0.0005, help='initial learning rate； default=0.001/0.0001')
    parser.add_argument('--step_size', default=20, help='decay step for lr decay, default:20/10')
    parser.add_argument('--lr_decay', default=0.7, help='decay rate for lr decay,default:0.2')

    parser.add_argument('--exp_decay', type=bool, default=False, help='exponentially decay, step by step')

    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--data_dir', type=str, default='modelnet40_ply_hdf5_2048')

    parser.add_argument('--random_seed', type=int, default=0, help='int or None')

    parser.add_argument('--trainable_tmp', type=bool, default=False)

    parser.add_argument('--pre_sample', type=str, default='fps')

    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--beta', type=float, default=0.01, help='default=0.01')
    parser.add_argument('--w_sigma', type=float, default=1, help='weights for sigma')

    parser.add_argument('--S_beta', type=float, default=0, help='beta in Simplify loss')
    parser.add_argument('--S_gamma', type=float, default=1)
    parser.add_argument('--S_delta', type=float, default=0)

    parser.add_argument('--lmbda', type=float, default=0.5, help='weight for task loss')

    parser.add_argument('--save_last', type=bool, default=True)

    parser.add_argument('--use_atten', type=bool, default=True)

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
            checkpoint = torch.load(str(source_checkpoints_dir) + '/best_model.pth', map_location=torch.device(cur_device))
            model_state_dict = checkpoint['model_state_dict']
            sampler_dict = {k: v for k, v in model_state_dict.items() if 'sampler' in k}
            model_state_dict = model.state_dict()
            model_state_dict.update(sampler_dict)
            model.load_state_dict(model_state_dict)

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

    task_model, start_epoch = load_checkpoints(task_model, checkpoints_dir, source_checkpoints_dir)

    sample_criterion = MODEL.get_sample_loss(alpha=args.alpha, beta=args.beta, S_beta=args.S_beta, S_gamma=args.S_gamma, S_delta=args.S_delta).to(cur_device)
    task_criterion = TaskLoss_cls().to(cur_device)

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

    best_acc = 0.0
    n_sample_list = np.array([16,32,64,128,256,512])
    n_input_list = np.linspace(800,2000,21).astype('int')

    Acc_list = []

    for epoch in range(start_epoch, args.epoch):
        task_model, optimizer = momentum_lr_adjusting(args, task_model, optimizer, epoch)
        task_model.sampler.train()
        # check_trainable_parameters(task_model)

        mean_loss = []
        train_mean_correct = []

        for dataset_id in range(len(train_dataset_list)):
            print('train_file %d' % dataset_id)
            data_loader = torch.utils.data.DataLoader(train_dataset_list[dataset_id], batch_size=args.batch_size, shuffle=True)
            for batch_id, (points, label) in tqdm(enumerate(data_loader), total=len(data_loader)):
                optimizer.zero_grad()

                points = torch.Tensor(points)
                # n_input = n_input_list[batch_id % len(n_input_list)]
                # n_input = np.random.choice(n_input_list)
                # k_neighbors = int(40 * (n_input/1024))
                # points = random_sampling(points, n_input)[0]
                k_neighbors = 40

                points, label = points.to(cur_device).float(), label.to(cur_device).long()

                # check_trainable_parameters(task_model)
                if args.use_atten:
                    n_sample = n_sample_list[batch_id % len(n_sample_list)]
                    # n_sample = np.random.choice(n_sample_list)
                else:
                    n_sample = args.n_sample

                if args.single_anchor:
                    n_sample = args.n_sample

                pred_subset, source_subset = task_model.sampler(points.transpose(2, 1), n_sample, k_neighbors)
                pred = task_model(pred_subset.transpose(2, 1))

                offset = pred_subset - source_subset
                sample_loss = sample_criterion(points, pred_subset, offset=offset)
                task_loss = task_criterion(pred, label)
                total_loss = args.lmbda*task_loss + sample_loss

                acc = pred2acc(pred, label)
                train_mean_correct.append(acc)

                total_loss.backward()
                mean_loss.append(total_loss.item())
                optimizer.step()

        log_string('epoch %d, train loss %.5f, acc %.5f' % (epoch, np.mean(mean_loss), np.mean(train_mean_correct)))


        with torch.no_grad():
            task_model.sampler.eval()
            mean_correct = []
            for dataset_id in range(len(test_dataset_list)):
                print('test_file %d' % dataset_id)
                data_loader = torch.utils.data.DataLoader(test_dataset_list[dataset_id], batch_size=args.batch_size, shuffle=False)
                for batch_id, (points, label) in tqdm(enumerate(data_loader), total=len(data_loader)):

                    # n_input = n_input_list[batch_id % len(n_input_list)]
                    # k_neighbors = int(40 * (n_input / 1024))
                    # points = points[:,:n_input,:]
                    k_neighbors=40
                    points, label = points.to(cur_device).float(), label.to(cur_device).long()

                    if args.use_atten:
                        # n_sample = n_sample_list[batch_id % len(n_sample_list)]
                        n_sample = np.random.choice(n_sample_list)
                    else:
                        n_sample = args.n_sample

                    if args.single_anchor:
                        n_sample = args.n_sample

                    pred_subset, source_subset = task_model.sampler(points.transpose(2,1), n_sample, k_neighbors)

                    pred = task_model(pred_subset.transpose(2,1))
                    acc = pred2acc(pred, label)
                    mean_correct.append(acc)

            test_mean_acc = np.mean(mean_correct)
            log_string('epoch %d, avg test acc %.5f' % (epoch, test_mean_acc))

        Acc_list.append(test_mean_acc)

        if test_mean_acc > best_acc:
            best_acc = test_mean_acc
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

    return best_acc


if __name__ == '__main__':
    args = parse_args()
    main(args)