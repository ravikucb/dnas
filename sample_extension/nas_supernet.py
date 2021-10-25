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
import torch.nn.functional as F
import numpy as np

# This class implements the OVERALL
# SuperNet class. For each experiment,
# we would create a subclass
# e.g. DLRMSuperNet.


class SuperNet(nn.Module):
    def __init__(self):
        # Superclas initialization.
        super(SuperNet, self).__init__()

        # self.theta_parameters should be
        # a torch.nn.ParameterList of
        # regular torch Parameters.
        # Each of the individual
        # Parameters will be a vector
        # with dimension equal to the
        # number of different inputs
        # over which to take a weighted average.
        self.theta_parameters = None

        # self.mask_values should be a
        # simple list of torch Tensors.
        # These values are generated
        # every time soft_sample or
        # hard_sample is called. We
        # do not want these to be
        # trained (and thus do not
        # want these to be parameters)
        # as we actually want to train
        # self.theta_parameters.
        self.mask_values = []

        # This is just the number of
        # mask lists so that we do not
        # have to re-calculate the size
        # in soft_sample and hard_sample;
        # it is set by the subclass in
        # the subclass __init__() method.
        self.num_mask_lists = None

        # Whether or not the
        # implementation has been checked.
        self.implementation_checked = False

        # Whether or not the mask values are fixed.
        self.mask_values_fixed = False

    # This function should be
    # overloaded in the subclass.
    def forward(self, inputs):
        assert False, "ERROR: YOU ARE \
                        CALLING THE DEFAULT \
                        forward() IMPLEMENTATION \
                        IN SuperNet WHICH \
                        MEANS YOU ARE NOT \
                        OVERLOADING IT. \
                        THIS IS A VERY \
                        BAD SIGN. PLEASE \
                        CHECK YOUR CODE \
                        FOR BUGS."

    # This function will be
    # called before every call to forward().
    def soft_sample(self, temp, batch_size):
        # If the mask values are fixed, do nothing.
        if self.mask_values_fixed is True:
            return

        # Basic input check.
        assert temp > 0.00, "ERROR: YOU ARE \
                                    PASSING A NON- \
                                    POSITIVE TEMPERATURE \
                                    INTO THE soft_sample \
                                    Gumbel Softmax IN \
                                    SuperNet."

        # Soft-sample the mask parameters
        # before every forward pass.
        for mask_list_index in range(self.num_mask_lists):
            # For each mask, take a soft
            # sample Gumbel Softmax.

            # We have to replicate the
            # theta parameters batch_size
            # times so that the architecture
            # weighted sum parameters can be
            # different for each example in
            # the batch.
            curr_theta_params = self.theta_parameters[mask_list_index]
            new_shape = (batch_size, -1)
            curr_theta_param_exp = curr_theta_params.expand(new_shape)

            # Set the mask values using F.gumbel_softmax.
            curr_mask = F.gumbel_softmax(curr_theta_param_exp, tau=temp)
            self.mask_values[mask_list_index] = curr_mask

    def hard_sample(self):
        # This results in the mask parameters
        # being one-hot in every mask
        # parameter list.

        # This function will return the
        # sampled mask values as a list
        # of Python lists. We use
        # torch.no_grad() because we do
        # not want to train based on
        # this sampling.

        # If the mask values
        # are fixed, do nothing.
        if self.mask_values_fixed is True:
            return

        with torch.no_grad():
            current_mask_values = []

            for mask_list_index in range(self.num_mask_lists):
                # For each mask, take
                # a hard-sample
                # Gumbel Softmax.

                # We have to reshape
                # the current theta
                # parameter to 2D
                # so that
                # F.gumbel_softmax works.
                curr_theta_params = self.theta_parameters[mask_list_index]
                curr_theta_param_exp = curr_theta_params.view(1, -1)

                # Take the 0th element
                # of F.gumbel_softmax's
                # output so that
                # self.mask_
                # values[mask_list_index]
                # is a 1D Tensor,
                # not 2D as the
                # output of
                # F.gumbel_softmax
                # will be.
                mask_one_hot = F.gumbel_softmax(curr_theta_param_exp,
                                                hard=True)
                one_mask = mask_one_hot[0]
                mask_list = list(one_mask.detach().cpu().numpy())
                current_mask_values.append(mask_list)

            return current_mask_values

    # This will be called every time we want
    # a sum of the input values weighted by
    # the mask values.
    def calculate_weighted_sum(self,
                                weights,
                                mats,
                                n_mats=3):
        """
        weights are the weights
        to apply to the input
        mats, which are the
        matrices to be
        weighted. n_mats
        is the length of
        matrices.
        """

        # Depending on the number of indices in mats,
        # a different expansion shape will be needed.
        mats_num_indices = len(list(mats[0].size()))
        expand_shape = (1,) + mats_num_indices * (-1,)

        # Combine each tensor into a single tensor with all the operator outputs..
        mat_list = [mats[i].expand(expand_shape) for i in range(n_mats)]
        all_mats = torch.cat(mat_list)

        # This operation moves the n_mats dimension to the end to allow for the BMM operation below.
        for transpose_index in range(mats_num_indices):
            all_mats = torch.transpose(all_mats, transpose_index, transpose_index + 1)

        all_mats_reshaped = all_mats

        # Adust the shape of the weights to allow for the BMM operation.
        weights_exp = weights.expand((1, -1, -1))
        weights_exp_t_0 = torch.transpose(weights_exp, 0, 2)
        weights_reshaped = torch.transpose(weights_exp_t_0, 0, 1)

        # Collapse all_mats_reshaped into 3D tensor to run BMM.
        reshape_all_mats_size = np.cumprod(list(mats[0].size())[1:])[-1]
        reshape_all_mats_batch_size = list(mats[0].size())[0]
        all_mats_reshaped = all_mats.reshape(reshape_all_mats_batch_size, reshape_all_mats_size, n_mats)

        # BMM operation - actual "weighted sum."
        result = torch.bmm(all_mats_reshaped, weights_reshaped)

        # Reshape back to original dimensions i.e. reverse tensor collapse operation.
        # (8, 32, 50, 50, 3) --> (8, 80000, 3) x (8, 3, 1) = (8, 80000, 1)[:, :, 0] = (8, 80000) --> (8, 32, 50, 50)
        reshape_result_dimensions = (reshape_all_mats_batch_size,) + tuple(list(mats[0].size())[1:])
        reshaped_result = result[:, :, 0].reshape(reshape_result_dimensions)

        return reshaped_result
