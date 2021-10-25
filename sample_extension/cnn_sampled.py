#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/24/21

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):
    """
    Sampled ConvSuperNet.
    """

    def __init__(self, conv1_op, conv2_op):
        """
        Creates CNN.
        """

        # Superclas initialization.
        super(ConvNet, self).__init__()

        # Store kernels.
        if conv1_op == 0: self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
        elif conv1_op == 1: self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, padding = 2)
        elif conv1_op == 2: self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, padding = 3)
        else: raise ValueError(f"Invalid conv1_op {conv1_op}.")

        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        if conv2_op == 0: self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        elif conv2_op == 1: self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2)
        elif conv2_op == 2: self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 7, padding = 3)
        else: raise ValueError(f"Invalid conv2_op {conv2_op}.")

        self.pool2 = nn.MaxPool2d(kernel_size = 5)

        self.classification_layer = nn.Linear(64 * 5 * 5, 5)

    def forward(self, input_images):
        """
        Runs the images through the network, producing classification outputs.
        """

        # Run the images through the CNN.

        # First convolutional layer.
        conv1_output = self.conv1(input_images)

        # First pooling layer - 2x2.
        pool1_output = self.pool1(conv1_output)

        # Second convolutional layer.
        conv2_output = self.conv2(pool1_output)

        # Second pooling layer - 5x5.
        pool2_output = self.pool2(conv2_output)

        # Flatten output for linear layer.
        flattened_output = pool2_output.reshape((-1, 64 * 5 * 5))

        # Linear layer to generate classification logits.
        classification_logits = self.classification_layer(flattened_output)

        # Return the logits directly - no softmax. This is what nn.CrossEntropyLoss expects.
        return classification_logits

if __name__ == "__main__":
    for conv1_op_index in [0, 1, 2]:
        for conv2_op_index in [0, 1, 2]:
            # Sampled architecture.
            sampled_arch = {"conv1_op" : conv1_op_index, "conv2_op" : conv2_op_index}

            # Create a convnet.
            convnet = ConvNet(**sampled_arch)

            # Move it to GPU.
            convnet.to("cuda:0")

            # Create a random batch of data.
            input_image_data = torch.randn(8, 3, 50, 50).to("cuda:0")

            # Get classification outputs.
            outputs = convnet(input_image_data)

            print(f"SAMPLED ARCH {sampled_arch}, INPUT IMAGES {input_image_data.size()}, OUTPUT {outputs.size()}.")
