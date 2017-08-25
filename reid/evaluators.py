from __future__ import print_function, absolute_import
import os
import matplotlib.pyplot as plt
import numpy as np

import time
from collections import OrderedDict
import torch

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels


def extract_bbox_features(model, data_loader, print_freq=1):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()  # not used yet

    end = time.time()
    for i, (imgs, file_names) in enumerate(data_loader):
        data_time.update(time.time() - end)

        # # save imgs after transform
        # for img, file_name in zip(imgs, file_names):
        #     npimg = img.numpy() * 255
        #     npimg = npimg.astype('uint8')
        #     imgn = np.transpose(npimg, (1, 2, 0))
        #     plt.imsave('', imgn)

        outputs = extract_cnn_feature(model, imgs)
        for file_name, output in zip(file_names, outputs):
            features[file_name] = output
        batch_time.update(time.time() - end)
        end = time.time()

        # if (i + 1) % print_freq == 0:
        #     print('Extract Features: [{}/{}]\t'
        #           'Time {:.3f} ({:.3f})\t'
        #           'Data {:.3f} ({:.3f})\t'
        #           .format(i + 1, len(data_loader),
        #                   batch_time.val, batch_time.avg,
        #                   data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        pow2 = torch.pow(x, 2).sum(1)
        dist = pow2.expand(n, n) + pow2.t().expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def probability(features):
    n = len(features)

    x = torch.cat(list(features.values()))
    x = x.view(n, -1)
    # |a| * |b|, vector with size n * n
    vector_len = torch.pow(x, 2).sum(1).sqrt()
    vector = torch.mm(vector_len, vector_len.t())
    # vector multiplication: a . b
    # vector_m with size n * n
    vector_m = torch.mm(x, x.t())
    # cos, range: [-1, 1]
    cos = vector_m / vector
    # (cos + 1) / 2, convert range in [0, 1]
    cos_ = (cos + 1) / 2
    # convert range to (0, 1)
    cos_[cos_ >= 1] = 1 - 1e-5
    cos_[cos_ <= 0] = 0 + 1e-5

    return cos_


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None):
        features, _ = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery)

    def bbox_evaluate(self, data_loader, use_cos, metric=None):
        features, _ = extract_bbox_features(self.model, data_loader)
        if use_cos == 'y':
            distmat = probability(features)
        else:
            distmat = pairwise_distance(features, None, None, metric=metric)
        return distmat

    def extract_proposal_features(self, data_loader):
        features, _ = extract_bbox_features(self.model, data_loader)
        n = len(features)

        features_ = torch.cat(list(features.values()))
        features_ = features_.view(n, -1)
        return features_
