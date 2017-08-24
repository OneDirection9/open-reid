#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: WarmerHan
# Time: 17-8-24 下午12:45
# File: pose_detection_folder.py
# Software: PyCharm

from __future__ import absolute_import
import os
import torch.utils.data as data

from PIL import Image

def key(x):
    return int(os.path.splitext(x)[0])

class PoseDetectionFolder(data.Dataset):
    def __init__(self, img_dir, input_transform=None, target_transform=None):
        super(PoseDetectionFolder, self).__init__()
        self.img_filenames = [os.path.join(img_dir, fname) for fname in sorted(os.listdir(img_dir), key=key)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        img_file = self.img_filenames[index]
        img = Image.open(img_file).convert('RGB')
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, os.path.basename(img_file).split('.')[0]