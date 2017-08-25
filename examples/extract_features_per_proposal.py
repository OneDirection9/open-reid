#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: WarmerHan
# Time: 17-8-25 上午10:52
# File: extract_features_per_proposal.py
# Software: PyCharm
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: WarmerHan
# Time: 17-8-23 上午10:45
# File: mytest.py
# Software: PyCharm

from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time
import scipy.io as scio
import codecs
import shutil

import numpy as np
import sys
from collections import OrderedDict
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from pose_detection_folder import PoseDetectionFolder


def bbox_data(img_dir, height, width, batch_size, workers):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    test_loader = DataLoader(PoseDetectionFolder(img_dir, input_transform=test_transformer), batch_size=batch_size,
                             num_workers=workers, shuffle=False, pin_memory=True)

    return test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
            (256, 128)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remote 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Extract features for each proposal:")
        data_dir = args.data_dir
        num_dirs = len(os.listdir(data_dir))
        for i, video_name in enumerate(os.listdir(data_dir)):
            video_proposals_dir = os.path.join(data_dir, video_name)
            if not os.path.isdir(video_proposals_dir):
                continue

            save_feature_dir = os.path.join(args.save_dir, video_name)
            if os.path.exists(save_feature_dir):
                shutil.rmtree(save_feature_dir)
            os.mkdir(save_feature_dir)

            num_frames = len(os.listdir(video_proposals_dir))
            for j, frame_name in enumerate(os.listdir(video_proposals_dir)):
                img_dir = os.path.join(video_proposals_dir, frame_name)
                if not os.path.isdir(img_dir):
                    continue

                start_time = time.time()
                test_loader = bbox_data(img_dir, args.height, args.width, args.batch_size, args.workers)
                features_ = evaluator.extract_proposal_features(test_loader)
                features_mat = to_numpy(features_)
                save_file = os.path.join(save_feature_dir, frame_name)
                scio.savemat(save_file, {'feature': features_mat})

                end_time = time.time()
                print('Calculate: {:10} [{:4}/{:<4}].\t'
                      'Frames: [{:4}/{:<4}]\t'
                      'Shape: {:<4} * {:4}\t'
                      'Time: {:.3f}'
                      .format(video_name, i + 1, num_dirs, j + 1, num_frames, *features_mat.shape,
                              end_time - start_time))
        return


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    parser.add_argument('--use-cos', type=str, default='y',
                        choices=['y', 'n'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--save-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'dets'))
    main(parser.parse_args())
