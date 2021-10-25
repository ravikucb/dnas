#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/25/21

# Import statements.
import torch
import torch.nn as nn
import numpy as np


class LinearDLRM(nn.Module):
    """
    This layer is essentially equivalent
    to a regular nn.Linear layer, except
    that it uses the weight initialization
    from dlrm_s_pytorch.py.
    """

    def __init__(self,
                in_feat,
                out_feat,
                bias=False):

        # Superclass initialization.
        super(LinearDLRM, self).__init__()

        # Store for later.
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias

        # Create the layer.
        self.linear_layer = nn.Linear(self.in_feat,
                                        self.out_feat,
                                        bias=self.bias)

        # Change initialization to match DLRM code
        # from dlrm_s_pytorch.py.
        with torch.no_grad():
            mean = 0.0

            std_dev = np.sqrt(2.0 / (float(self.in_feat) + float(self.out_feat)))

            W = np.random.normal(mean,
                                std_dev,
                                size=(self.out_feat, self.in_feat)).astype(np.float32)

            std_dev = np.sqrt(1.0 / float(self.out_feat))

            bt = np.random.normal(mean,
                                std_dev,
                                size=self.out_feat).astype(np.float32)

            self.linear_layer.weight.data = torch.tensor(W, requires_grad=True)

            if self.bias is True:
                self.linear_layer.bias.data = torch.tensor(bt, requires_grad=True)

    def forward(self, input_vectors):
        """
        Run input_vectors through the layer.
        """

        return self.linear_layer(input_vectors)
