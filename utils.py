#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Ravi Krishna 07/25/21

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
KAGGLE_DAYS = 7

def vectors_sum(vecs):
    """
    Sum vectors in a list.
    """

    cat_vecs = torch.cat(vecs, -1)
    cat_vecs_2d = torch.reshape(cat_vecs, (len(vecs), vecs[0].size()[0]))
    vecs_sum = torch.sum(cat_vecs_2d, axis=0)
    return vecs_sum

def calculate_latency_efficient(mask_vals,
                                indices_look_up_table,
                                latency_table_ms,
                                num_s_net_layers,
                                fc_sizes_list,
                                input_dim,
                                output_dim):
    """
    Calculates the overall super net latency
    (across each input in the batch).
    """

    # Store the total latency as a vector of
    # dim batch_size.  We can get
    # batch_size from the dim of the mask values.
    curr_batch_s = list(mask_vals[0][0].size())[0]
    layer_latencies = []

    # We need to store the input size probs
    # for the next layer as we go through the super
    # net (next layer means that one we just
    # calculated the latency for, not the one we are
    # going to calculate the latency for next).
    next_input_probs = torch.ones(curr_batch_s, 1).to(latency_table_ms.device)

    # Calculate the latency for each layer in the
    # super net, starting from the last layer.
    for layer_ix in range(num_s_net_layers - 1, -1, -1):

        # Calculate latency for each output size
        # and then take weighted sum with the
        # output size probs.
        curr_layer_latencies = []

        if layer_ix != (num_s_net_layers - 1):
            fc_sizes_use = fc_sizes_list
        else:
            fc_sizes_use = [output_dim]

        for o_size_ix, curr_o_size in enumerate(fc_sizes_use):
            # Store the latency for all of the operators in
            # this layer with the output size curr_o_size.
            curr_o_size_latencies = []

            if layer_ix != 0:
                fc_sizes_use_i = fc_sizes_list
            else:
                fc_sizes_use_i = [input_dim]

            for i_size_ix, curr_i_size in enumerate(fc_sizes_use_i):
                # Calculate the latency for this operator.

                # Get the string corresponding to this FC op.
                look_up_str = f"fc_{curr_i_size}_{curr_o_size}"

                # Get the index of the look-up table corresponding
                # to the operator string.
                curr_op_latency_table_index = indices_look_up_table[look_up_str]

                # Get the latency from the latency table at that index.
                curr_op_latency = latency_table_ms[curr_op_latency_table_index]

                # Get the probs for this layer.
                curr_op_probs = mask_vals[layer_ix][o_size_ix][:, i_size_ix]

                # Add to curr_o_size_latency.
                curr_o_size_latencies.append(curr_op_latency * curr_op_probs)

            # Now get the probs for this output size.
            # next_input_probs is of size (batch_size, num_input_sizes_for_next_layer).
            curr_o_size_probs = next_input_probs[:, o_size_ix]

            # Sum the curr_o_size_latencies.
            curr_o_size_latency = vectors_sum(curr_o_size_latencies)

            # Add the weighted latency to the overall latency of this layer.
            weighted_latency = torch.mul(curr_o_size_latency, curr_o_size_probs)
            curr_layer_latencies.append(weighted_latency)

        # Update next_input_probs.
        # If layer_ix is 0, this will not matter anyway.
        old_next_input_probs = next_input_probs

        # Input and output sizes.
        if layer_ix != 0:
            curr_i_sizes = fc_sizes_list
        else:
            curr_i_sizes = [input_dim]

        if layer_ix != (num_s_net_layers - 1):
            curr_o_sizes = fc_sizes_list
        else:
            curr_o_sizes = [output_dim]

        # Input probabilities.
        table_device = latency_table_ms.device
        zero_vector = [0.00] * len(curr_i_sizes)
        next_input_probs = torch.zeros(curr_batch_s, len(curr_i_sizes)).to(table_device)

        for curr_i_size_ix, _curr_i_size in enumerate(curr_i_sizes):
            # The probability that we will use this input size
            # is the sum of the probs that we will use any
            # of its output sizes times the correct probs.
            curr_i_size_probs_list = []
            for curr_o_size_ix, _curr_o_size in enumerate(curr_o_sizes):
                # Mask values.
                curr_mask_vals = mask_vals[layer_ix][curr_o_size_ix][:, curr_i_size_ix]

                # Old input probs.
                curr_old_next_probs = old_next_input_probs[:, curr_o_size_ix]

                # Probabilities.
                curr_probs = torch.mul(curr_mask_vals, curr_old_next_probs)
                curr_i_size_probs_list.append(curr_probs)

            # Sum curr_i_size_probs_list to get curr_i_size_probs.
            curr_i_size_probs = vectors_sum(curr_i_size_probs_list)

            # Set the correct values in next_input_probs to curr_i_size_probs.
            next_input_probs[:, curr_i_size_ix] = curr_i_size_probs

        # Sum the curr_layer_latencies to get current_layer_latency.
        current_layer_latency = vectors_sum(curr_layer_latencies)

        # Add the latency for the current layer to the overall super net latency.
        layer_latencies.append(current_layer_latency)


    # Calculate the total latency as the sum of each layer's latency.
    total_latency = vectors_sum(layer_latencies)

    # Return the total latency.
    return total_latency


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
