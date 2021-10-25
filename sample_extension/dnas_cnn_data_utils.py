#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/14/21

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Dataset lengths.
dataset_lengths = {"train-weights" : 7200, "train-archparams" : 1800, "train" : 10000, "val" : 500, "test" : 500}

class CNNDataset(Dataset):
    def __init__(self, dataset_split_type):
        """
        Initializes dataset
        dataset_split_type must be one of: train-weights
                                           train-archparams
                                           val
                                           test
        """

        # Superclass initialization.
        super(CNNDataset, self).__init__()

        # Store for later.
        self.dataset_split_type = dataset_split_type
        self.length = dataset_lengths[self.dataset_split_type]
        self.num_classes = 5

    def __len__(self):
        """
        Returns dataset length.
        """

        return self.length

    def __getitem__(self, idx):
        """
        Returns the item (i.e. image) in the dataset
        at index idx and the label for that item. For
        an actual implementation, this would fetch and
        load and image, but for this example we just
        generate random data.
        """

        return (torch.ones((3, 50, 50)).to(torch.float32) * 0.01 * (idx % 100), torch.tensor(idx % self.num_classes))
