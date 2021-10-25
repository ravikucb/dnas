#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/24/21

# LR step function based heavily on existing
# PyTorch implementations of LR schedulers,
# and specifically _LRScheduler and LambdaLR.

# Import statements.
import torch
import torch.optim

# Constants.
STR_TO_OPTIM = {"sgd" : torch.optim.SGD,
                "adam" : torch.optim.Adam,
                "adagrad" : torch.optim.Adagrad}

# Adjusts optimizer learning rate.
def step_lambda_lr(optimizer,
                    lr_lambdas,
                    current_epoch,
                    initial_optimizer_lrs):
    # Iterate through each parameter group
    # in the optimizer, as well as the
    # lambda and the original LR.
    lists_zip_obj = zip(optimizer.param_groups, lr_lambdas, initial_optimizer_lrs)
    for param_group, lr_lambda, initial_lr in lists_zip_obj:
        param_group["lr"] = lr_lambda(current_epoch) * initial_lr

    # Add a return statement for clarity.
    return

def arch_sampling_str_to_dict(sampling_str):
    """
    Converts string representations of architecutre
    sampling methodology to the dictionary format
    that nas_searchmanager.SearchManager uses.

    Example:
    \"1:4,2:4,3:4,4:4\" is converted to
    {1.0 : 4, 2.0 : 4, 3.0 : 4, 4.0 : 4}.
    """

    # Remove any spaces that the user may have
    # entered into the string representation.
    sampling_str = sampling_str.replace(" ", "")

    # Split by comma to get the architectures sampled
    # at each number of architecture parameters epochs
    # completed.
    archs_sampled_list = sampling_str.split(",")

    # Convert the list of epoch:num_archs_to_sample
    # strings to the correct dictionary format.
    arch_sampling_dict = {}
    for curr_arch_sampling in archs_sampled_list:
        # Split by colon.
        [num_epochs_elapsed, num_archs_to_sample] = curr_arch_sampling.split(":")

        # Add to dictionary.
        arch_sampling_dict[float(num_epochs_elapsed)] = int(num_archs_to_sample)

    # Return the dictionary.
    return arch_sampling_dict
