#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/14/21

import torch
import torch.nn as nn
import torch.nn.functional as F
from nas_supernet import SuperNet
import numpy as np

class ConvSuperNet(SuperNet):
    """
    DNAS supernet for convolutional neural network.

    This search space is purely an EXAMPLE.
    """

    def __init__(self):
        """
        Creates CNN supernet with the structure defined in sample_extension/README_SAMPLE.md.
        """

        # Superclas initialization.
        super(ConvSuperNet, self).__init__()

        # Store kernels.
        self.conv1_choice1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
        self.conv1_choice2 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5, padding = 2)
        self.conv1_choice3 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, padding = 3)

        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2_choice1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv2_choice2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2)
        self.conv2_choice3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 7, padding = 3)

        self.pool2 = nn.MaxPool2d(kernel_size = 5)

        self.classification_layer = nn.Linear(64 * 5 * 5, 5)

        # Create other parameters.
        self.theta_parameters = nn.ParameterList([nn.Parameter(torch.tensor([0.00, 0.00, 0.00]), requires_grad = True), nn.Parameter(torch.tensor([0.00, 0.00, 0.00]), requires_grad = True)])
        self.mask_values = [None] * len(self.theta_parameters)
        self.num_mask_lists = len(self.mask_values)

        # Store current cost.
        self.curr_cost = None

        # Costs for convolutional operators.
        self.conv1_costs = torch.tensor([1.0, 2.0, 3.0])
        self.conv2_costs = torch.tensor([2.0, 4.0, 6.0])

    def calculate_cost(self):
        """
        This is meant to demonstrate that arbitraty HW cost functions can be used.
        """

        # Get mask values for both supernet layers.
        conv1_mask_values = self.mask_values[0]
        conv2_mask_values = self.mask_values[1]

        # Get the weighted average cost.
        weighted_avg_cost = torch.matmul(conv1_mask_values, self.conv1_costs) + torch.matmul(conv2_mask_values, self.conv2_costs)

        # Return cost.
        return weighted_avg_cost

    def to(self, device):
        """
        Overrides the original to() (nn.Module.to) and also moves self.theta_parameters,
        self.conv1_costs, and self.conv2_costs.
        """

        nn.Module.to(self, device)
        self.theta_parameters = self.theta_parameters.to(device)
        self.conv1_costs = self.conv1_costs.to(device)
        self.conv2_costs = self.conv2_costs.to(device)

    def forward(self, input_images, sampling="None", temperature=-1.0):
        """
        Runs the images through the network, producing classification outputs.
        Performs any supernet sampling as needed depending on input arguments.
        """

        # Get the current batch size.
        curr_batch_size = int(list(input_images.size())[0])

        # Sample if needed.
        if sampling == "soft":
            self.soft_sample(temperature, curr_batch_size)

        # Calculate the current cost of the supernet.
        self.curr_cost = self.calculate_cost()

        # Run the images through the CNN.

        # First convolutional layer.
        conv1_outputs = [conv1_op(input_images) for conv1_op in [self.conv1_choice1, self.conv1_choice2, self.conv1_choice3]]
        conv1_avg_output = self.calculate_weighted_sum(self.mask_values[0], conv1_outputs, n_mats = 3)

        # First pooling layer - 2x2.
        pool1_output = self.pool1(conv1_avg_output)

        # Second convolutional layer.
        conv2_outputs = [conv2_op(pool1_output) for conv2_op in [self.conv2_choice1, self.conv2_choice2, self.conv2_choice3]]
        conv2_avg_output = self.calculate_weighted_sum(self.mask_values[1], conv2_outputs, n_mats = 3)

        # Second pooling layer - 5x5.
        pool2_output = self.pool2(conv2_avg_output)

        # Flatten output for linear layer.
        flattened_output = pool2_output.reshape((-1, 64 * 5 * 5))

        # Linear layer to generate classification logits.
        classification_logits = self.classification_layer(flattened_output)

        # Return the logits directly - no softmax. This is what nn.CrossEntropyLoss expects.
        return classification_logits

    def sample_arch(self):
        """
        Returns a configuration which can be used to initialize
        a sampled architecture.
        """

        # Get sampled mask values.
        sampled_mask_values = self.hard_sample()

        # Find which operators to use.
        conv1_op = np.argmax(sampled_mask_values[0])
        conv2_op = np.argmax(sampled_mask_values[1])

        # Return the sampled architecture.
        return {"conv1_op" : conv1_op, "conv2_op" : conv2_op}

if __name__ == "__main__":
    # Create a supernet.
    supernet = ConvSuperNet()

    # Move it to GPU.
    supernet.to("cuda:0")

    # Create a random batch of data.
    input_image_data = torch.randn(8, 3, 50, 50).to("cuda:0")
    sampling_type = "soft"
    temperature = 0.1

    # Get classification outputs.
    outputs = supernet(input_image_data, sampling = sampling_type, temperature = temperature)

    # Sample architecture.
    for i in range(1000):
        sampled_arch = supernet.sample_arch()
        print(f"SAMPLED ARCHITECTURE {i}: {sampled_arch}")
