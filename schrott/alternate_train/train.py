import argparse
import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from alternate_train.batch_engine import valid_trainer, batch_trainer
from alternate_train.dataset.AttrDataset import AttrDataset, get_transform, AttributesDataset
from alternate_train.loss.CE_loss import CEL_Sigmoid
from alternate_train.models.base_block import FeatClassifier, BaseClassifier
from alternate_train.models.resnet import resnet50
from alternate_train.tools.function import get_model_log_path, get_pedestrian_metrics
from alternate_train.tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed

set_seed(605)


def argument_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument("--debug", action='store_false')

    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--train_epoch", type=int, default=100)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr_ft", type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument("--lr_new", type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument('--device', default=0, type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')

    return parser


def main(args):
    dataset_dir = '/dataset'  # args.dataset
    exp_dir = os.path.join('exp_result', dataset_dir)
    model_dir, log_dir = get_model_log_path(exp_dir, dataset_dir)
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')
    batchsize = 32

    print('-' * 60)
    # print(f'train set: {dataset_dir} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(256, 192)  # ??????????????????????????
    print(train_tsfm)

    attributes = AttributesDataset(dataset=dataset_dir)

    train_set = AttrDataset(data_path=dataset_dir, csv_filename='train.csv',
                            attributes=attributes, transform=train_tsfm)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_set = AttrDataset(data_path=dataset_dir, csv_filename='val.csv',
                            attributes=attributes, transform=train_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


    labels = train_set.label
    sample_weight = labels.mean(0)

    backbone = resnet50()
    classifier = BaseClassifier(nattr=attributes.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    criterion = CEL_Sigmoid(sample_weight)

    param_groups = [{'params': model.module.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.module.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path)

    print(f'{dataset_dir},  best_metrc : {best_metric} in epoch{epoch}')


def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    for i in range(epoch):

        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        lr_scheduler.step(metrics=valid_loss, epoch=i)

        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)

        cur_metric = valid_result.ma

        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)

        result_list[i] = [train_result, valid_result]

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
