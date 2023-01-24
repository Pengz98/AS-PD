from utils.modelnet_data_loading import ModelNet40_h5_dataset_list
import importlib
import argparse
import logging
import datetime
from pathlib import Path
from tensorboardX import SummaryWriter
import torch.nn as nn
import os
import sys
import torch
import shutil
from tqdm import tqdm
import numpy as np
from utils import pointnet_provider as provider
from utils.large_modelnet_data_loading import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser(description='pointnet_cls')
    parser.add_argument('--model', type=str, default='pointnet_cls_vanilla')
    parser.add_argument('--log_dir', type=str, default='test', help='log path')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', default=250)
    parser.add_argument('--n_point', type=int, default=2048)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', default=0.001, help='initial learning rate； default=0.001')
    parser.add_argument('--step_size', default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', default=0.7, help='decay rate for lr decay')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--data_dir', type=str, default='modelnet40_ply_hdf5_2048')

    parser.add_argument('--normalize', type=str, default=False)

    parser.add_argument('--use_uniform_sample', type=bool, default=False)

    parser.add_argument('--use_large_dataset', type=bool, default=False)

    parser.add_argument('--process_data', type=bool, default=True)
    return parser.parse_args()


def data_loading(args):
    '''data loading'''
    train_dataset_list = ModelNet40_h5_dataset_list(dataset_name=args.data_dir, split='train', n_point=args.n_point, normalize=args.normalize)
    test_dataset_list = ModelNet40_h5_dataset_list(dataset_name=args.data_dir, split='test', n_point=args.n_point, normalize=args.normalize)
    num_train_data = 0
    num_test_data = 0
    for loader_id in range(len(train_dataset_list)):
        num_train_data += len(train_dataset_list[loader_id])
    for loader_id in range(len(test_dataset_list)):
        num_test_data += len(test_dataset_list[loader_id])
    print('num of train_data: %d' % num_train_data)
    print('num of test_data: %d' % num_test_data)
    return train_dataset_list, test_dataset_list


def model_loading(model_name):
    '''models loading '''
    MODEL = importlib.import_module(model_name)
    model = MODEL.get_model(return_global_feat=False).to(cur_device)
    try:
        criterion = MODEL.get_loss().to(cur_device)
    except:
        criterion = None
    return model, criterion


def train(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''create dir'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.model)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(os.path.join(exp_dir, 'lossvisualization'))

    '''log'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''data loading'''
    print('data loading')
    if args.use_large_dataset:
        # large scale modelnet40
        train_dataset_list = [ModelNetDataLoader(args, split='train', process_data=args.process_data)]
        test_dataset_list = [ModelNetDataLoader(args, split='test', process_data=args.process_data)]
    else:
        train_dataset_list, test_dataset_list = data_loading(args)
    num_train_data = 0
    num_test_data = 0
    for loader_id in range(len(train_dataset_list)):
        num_train_data += len(train_dataset_list[loader_id])
    for loader_id in range(len(test_dataset_list)):
        num_test_data += len(test_dataset_list[loader_id])
    log_string('num of train_data: %d' % num_train_data)
    log_string('num of test_data: %d' % num_test_data)

    '''models loading'''
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(return_global_feat=False).to(cur_device)
    criterion = MODEL.get_loss().to(cur_device)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet_utils.py', str(exp_dir))


    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pre-trained models')
    except:
        log_string('No existing models, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            # classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_instance_acc = 0.0
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch, epoch, args.epoch - 1))

        '''adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        classifier = classifier.train()
        mean_loss = []
        mean_correct = []

        '''learning one epoch'''
        for dataset_id in range(len(train_dataset_list)):
            print('train_file %d' % dataset_id)
            data_loader = torch.utils.data.DataLoader(train_dataset_list[dataset_id], batch_size=args.batch_size, shuffle=True)
            for batch_id, (points, label) in tqdm(enumerate(data_loader), total=len(data_loader)):

                optimizer.zero_grad()

                points = points.data.numpy()
                points[:, :, 0:3] = provider.rotate_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)

                cur_batch_size, n_point, _ = points.size()
                points, label = \
                    points.float().to(cur_device), label.long().to(cur_device)
                points = points.transpose(2, 1)

                pred = classifier(points)

                cls_loss = criterion(pred, label)

                from train_sampler import gpu_memory_check
                total, used, free = gpu_memory_check()
                print('gpu memory cost for all:')
                print(total, used, free)

                cls_loss.backward()
                mean_loss.append(cls_loss.item())
                optimizer.step()

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(label.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(cur_batch_size))

        train_instance_acc = np.mean(mean_correct)
        train_instance_loss = np.mean(mean_loss)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string('Train Instance Loss: %f' % train_instance_loss)

        with torch.no_grad():
            test_mean_correct = []
            classifier = classifier.eval()

            for dataset_id in range(len(test_dataset_list)):
                print('test_file %d' % dataset_id)
                data_loader = torch.utils.data.DataLoader(test_dataset_list[dataset_id], batch_size=args.batch_size)
                for batch_id, (points, label) in tqdm(enumerate(data_loader),total=len(data_loader)):
                    cur_batch_size, n_point, _ = points.size()
                    points, label = \
                        points.float().to(cur_device), label.long().to(cur_device)
                    points = points.transpose(2, 1)

                    pred = classifier(points)

                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(label.long().data).cpu().sum()
                    test_mean_correct.append(correct.item() / float(cur_batch_size))
            test_instance_acc = np.mean(test_mean_correct)
            log_string('Test Instance Accuracy: %f' % test_instance_acc)
            writer.add_scalar('Test_Instance_Acc', test_instance_acc, epoch)

            if (test_instance_acc >= best_instance_acc):
                best_instance_acc = test_instance_acc
                best_epoch = epoch + 1
            log_string('Best Instance Accuracy: %f' % (best_instance_acc))

            if (test_instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': test_instance_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1



if __name__ == '__main__':
    args = parse_args()
    train(args)
