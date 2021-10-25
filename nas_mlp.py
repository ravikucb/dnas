#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/25/21

# This super net is a regular MLP search super net,
# which allows for an arbitrary input and output dim.

# Import statements.
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calculate_latency_efficient
from nas_supernet import SuperNet
from nas_linear import LinearDLRM
import numpy as np

class SuperNetMLP(SuperNet):
    def __init__(self,
                input_dim,
                output_dim,
                num_s_net_layers,
                fc_sizes_list,
                indices_look_up_table,
                cost_table,
                relu_last_layer=True,
                batch_norm=True,
                last_layer_batch_norm=False,
                bias=False,
                last_layer_bias=True,
                last_layer_activation=F.relu):

        """
        Creates the SuperNetMLP.

        input_dim is the dim
        of the input feature vector to the MLP.

        output_dim is the dim of
        the output of ths MLP (i.e. the
        output size of the last FC layer).

        num_s_net_layers is the number of
        layers in the super net. For example,
        if num_s_net_layers == 1, then we
        would just have one FC layer with input
        dim input_dim and output
        dim output_dim.

        fc_sizes_list is the list of FC layer
        sizes that are to be searched.

        indices_look_up_table is a dictionary
        that takes an op name as input
        and returns its cost LUT index.

        Ths index (ix) returned by
        indices_look_up_table is the
        ix at which the cost (any cost e.g.
        latency, power) of that op is.

        cost_table is a PyTorch vector with the cost
        values for all of the operators (FC)
        in the MLP search super net. In other words,
        this is the cost look up table for each
        operator.

        If relu_last_layer == True, then we will use a
        relu for the output of the last layer; if it is
        False, we will not (e.g. relu_last_layer would
        be False in the case of an output that will
        have a sigmoid applied to it).

        Other parameters should be reasonable self-explanatory.
        """

        # Superclass initialization.
        super(SuperNetMLP, self).__init__()

        # Store for later.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_s_net_layers = num_s_net_layers
        self.fc_sizes_list = fc_sizes_list
        self.indices_look_up_table = indices_look_up_table
        self.cost_table = cost_table
        self.relu_last_layer = relu_last_layer
        self.bias = bias
        self.last_layer_bias = last_layer_bias
        self.last_layer_activation = last_layer_activation

        # Create all of the FC layers.
        self.fc_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])
        self.use_relu = []

        # Create the architecture parameters and mask values.
        self.theta_parameters = nn.ParameterList([])
        self.arch_param_indices = {}
        curr_arch_param_ix = 0

        for s_net_layer in range(self.num_s_net_layers):
            # Store the FC and batch norm layers
            # for this super net layer.
            curr_layer_fc_layers = nn.ModuleList([])
            curr_layer_bn_layers = nn.ModuleList([])

            # Decide the input and output sizes,
            # as well as whether or not to use a bias.
            # The FC size -1 denoted a skipped layer. Note that
            # we can only skip layers that are not
            # connected to either the input or output.
            # This is because if we skipped the first layer,
            # we would need to add an option to maintain
            # the input dimension - this may be added later
            # but would complicate the supernet structure.
            # Skipping the output layer isn't feasible unless
            # the output dimension is added as an MLP size option
            # and is required to be selected at the last layer.
            if s_net_layer == 0:
                curr_in_sizes = [self.input_dim]
                curr_o_sizes = self.fc_sizes_list
                curr_use_bias = self.bias
            elif s_net_layer == (self.num_s_net_layers - 1):
                curr_in_sizes = self.fc_sizes_list
                curr_o_sizes = [self.output_dim]
                curr_use_bias = self.last_layer_bias
            else:
                curr_in_sizes = self.fc_sizes_list + [-1]
                curr_o_sizes = self.fc_sizes_list
                curr_use_bias = self.bias

            # Create all of the operators for each output size.
            for curr_o_size_ix, curr_o_size in enumerate(curr_o_sizes):
                # Store the FC and batch norm layers for this output size.
                curr_o_size_fc_layers = nn.ModuleList([])
                curr_o_size_bn_layers = nn.ModuleList([])

                # Create all of the operators for each input size.
                # Need to allow for skipping an FC layer.
                for curr_in_size in (curr_in_sizes):
                    # Use nn.Identity() operators for the skipped layers.
                    if curr_in_size == -1:
                        curr_fc_layer = nn.Identity()

                    else:
                        # Create the correct FC and batch norm layers.
                        curr_fc_layer = LinearDLRM(curr_in_size,
                                                    curr_o_size,
                                                    bias=curr_use_bias)

                    # Append these operators to curr_o_size_fc_layers
                    # and curr_o_size_bn_layers.
                    curr_o_size_fc_layers.append(curr_fc_layer)

                # Append the ops for this output size to
                # curr_layer_fc_layers and curr_layer_bn_layers.
                # Append the curr architecture parameter to the
                # architecture parameters.
                curr_layer_fc_layers.append(curr_o_size_fc_layers)

                # self.theta_parameters must be a regular list
                # of parameters and cannot be a list of a list
                # of parameters. To prevent the implementation
                # from becoming confusing, we will store a
                # dictionary with the indices of the
                # architecture parameters based on the super
                # net layer ix and the output size ix.
                zero_theta_vec = [0.00] * len(curr_in_sizes)
                zero_theta_params = nn.Parameter(torch.Tensor(zero_theta_vec), requires_grad=True)
                self.theta_parameters.append(zero_theta_params)

                curr_param_str = str(s_net_layer) + "," + str(curr_o_size_ix)
                self.arch_param_indices[curr_param_str] = curr_arch_param_ix
                curr_arch_param_ix += 1

            # Append all of the layers for this super net layer
            # to self.fc_layers and self.bn_layers.
            self.fc_layers.append(curr_layer_fc_layers)

            # Decide whether or not to use relu.
            if s_net_layer == (self.num_s_net_layers - 1):
                self.use_relu.append(self.relu_last_layer)
            else:
                self.use_relu.append(True)

        # Create the mask values.
        self.mask_values = [None] * len(self.theta_parameters)
        self.num_mask_lists = len(self.mask_values)

        # For compatibility with DLRM, store the current cost
        # instead of returning it after each call to forward().
        self.curr_cost = None

    def calculate_cost(self):
        """
        Calculates the network cost by calling an
        external function with the correct mask values.
        """

        mask_values_correct_lists = []
        for curr_layer in range(self.num_s_net_layers):
            curr_mask_values = []

            # Range of output sizes.
            if curr_layer == (self.num_s_net_layers - 1):
                o_size_ix_range = 1
            else:
                o_size_ix_range = len(self.fc_sizes_list)

            for curr_o_size_ix in range(o_size_ix_range):
                curr_arch_param_str = str(curr_layer) + "," + str(curr_o_size_ix)
                curr_arch_ix = self.arch_param_indices[curr_arch_param_str]
                curr_mask_values.append(self.mask_values[curr_arch_ix])

            mask_values_correct_lists.append(curr_mask_values)

        # Calculate the cost of the entire super net,
        # using the current Gumbel Softmax operator
        # probabilities. This calculates the expected
        # latency of the whole super net with the current
        # theta parameters.
        return calculate_latency_efficient(mask_values_correct_lists,
                        self.indices_look_up_table,
                        self.cost_table,
                        self.num_s_net_layers,
                        self.fc_sizes_list,
                        self.input_dim,
                        self.output_dim)

    def forward(self, in_vecs, sampling="None", temperature=-1.0):
        """
        Runs in_vecs through the super net.

        Return both the output of the super net
        and the cost of the curr super net.
        """

        # Get the batch size.
        curr_batch_size = int(list(in_vecs.size())[0])

        # Run the sampling if necessary.
        if sampling == "soft":
            # This is where we soft-sample the super
            # net i.e. where we run the Gumbel Softmax
            # over the theta parameters to result
            # in the mask values.
            self.soft_sample(temperature, curr_batch_size)

        # Calculate the cost of the network.
        self.curr_cost = self.calculate_cost()

        # Run in_vecs through the super net.
        curr_layer_inputs = [in_vecs]

        # Net already sampled,
        # so now just run the
        # forward pass using those
        # mask values.
        for s_net_layer in range(self.num_s_net_layers):
            # Decide the curr output sizes and
            # store the inputs for the next layer.
            if s_net_layer != (self.num_s_net_layers - 1):
                curr_o_sizes = self.fc_sizes_list
            else:
                curr_o_sizes = [self.output_dim]

            new_curr_layer_inputs = []

            # Calculate the outputs for all of the output sizes.
            for curr_o_size_ix, curr_o_size in enumerate(curr_o_sizes):
                # Get the input sizes and store
                # the outputs of all of the
                # layers for this input size.
                if s_net_layer != 0:
                    if s_net_layer != (self.num_s_net_layers - 1):
                        curr_in_sizes = list(self.fc_sizes_list) + [-1]
                    else:
                        curr_in_sizes = self.fc_sizes_list
                else:
                    curr_in_sizes = [self.input_dim]

                # Outputs for all ops
                # for the current output
                # size.
                curr_o_size_all_layers_out = []

                # Calculate the output for each input.
                for curr_in_size_ix, curr_in_size in enumerate(curr_in_sizes):
                    # Select the current
                    # FC and BN layers.

                    # All ops in this
                    # super net layer.
                    all_fc = self.fc_layers[s_net_layer]

                    # Get specific ops.
                    curr_fc_layer = all_fc[curr_o_size_ix][curr_in_size_ix]

                    if self.use_relu[s_net_layer] is True:
                        curr_relu = F.relu
                    else:
                        if s_net_layer != (self.num_s_net_layers - 1):
                            curr_relu = nn.Identity()
                        else:
                            curr_relu = self.last_layer_activation

                    # If this is a skip layer op, then there is no actual
                    # input size for -1; rather, we need to select the input
                    # with the same size as the output.
                    if curr_in_size == -1:
                        curr_fc_out = curr_fc_layer(curr_layer_inputs[curr_in_sizes.index(curr_o_size)])
                    else:
                        curr_fc_out = curr_fc_layer(curr_layer_inputs[curr_in_size_ix])

                    layer_out = curr_relu(curr_fc_out)
                    curr_o_size_all_layers_out.append(layer_out)

                # Take the weighted sum over all of the outputs for this output size.
                curr_arch_str = str(s_net_layer) + "," + str(curr_o_size_ix)
                curr_arch_param_ixs = self.arch_param_indices[curr_arch_str]
                curr_mask_vals = self.mask_values[curr_arch_param_ixs]
                new_curr_layer_inputs.append(self.calculate_weighted_sum(curr_mask_vals,
                                curr_o_size_all_layers_out,
                                n_mats=len(curr_in_sizes)))

            # Set curr_layer_inputs to new_curr_layer_inputs
            # since we have calculated all of the outputs
            # for this layer.
            curr_layer_inputs = new_curr_layer_inputs

        # curr_layer_inputs should now have the outputs
        # for the network and it should have length 1.
        network_outputs = curr_layer_inputs[0]

        # Return the network outputs as well as the cost.
        return network_outputs

    def sample_mlp_arch(self):
        """
        Hard-samples from the Gumbel Softmax distribution
        and returns the resulting MLP configuration
        in DLRM format such that the list of sizes returned
        can be directly bassed as an ln_bot or ln_top argument
        to DLRM_Net.__init__().
        """

        # First, hard-sample from the distribution.
        sampled_mask_values = self.hard_sample()

        # Now, we need to determine the correct MLP sizes
        # based on the hard-sampled mask values.

        # Maps mask values for a particular
        # output size at a particular supernet
        # layer to mask values index.
        arch_params_indices = {}

        # Store the index to look up
        # in the look up table given
        # the current architecture
        # string.
        curr_arch_param_ix = 0

        # Get self.arch_params_indices.
        for s_net_layer in range(self.num_s_net_layers):
            # If this is not the last layer, then
            # the output dimension may be any of the
            # possible sizes; if it is, it can only
            # be the output dimension.
            if s_net_layer != (self.num_s_net_layers - 1):
                curr_out_sizes = self.fc_sizes_list
            else:
                curr_out_sizes = [self.output_dim]

            # Create all of the layers for each output size.
            # Used to be current_output_size instead of _.
            for current_output_size_index, _ in enumerate(curr_out_sizes):
                # self.architecture_parameters must be a
                # regular list of parameters and cannot
                # be a list of a list of parameters.
                # To prevent the implementation from
                # becoming confusing, we will store a
                # dictionary with the indices of the
                # architecture parameters based on
                # the super net layer index and
                # the output size index.
                curr_arch_str_key = str(s_net_layer) \
                    + "," + str(current_output_size_index)
                arch_params_indices[curr_arch_str_key] = curr_arch_param_ix
                curr_arch_param_ix += 1

        # Start from the last FC layer
        # and get all of the FC layer
        # sizes.
        previous_input_index = 0
        fc_sizes = []

        # Running the loop for s_net_layer = 0
        # will not actually do anything since
        # there is no FC size at the 0th layer
        # (this is just the input dimension).
        for s_net_layer in range(self.num_s_net_layers - 1, 0, -1):
            # Indices with max
            # maxk value for
            # each output size.
            o_size_max_ixs = []

            # Set the current FC sizes.
            if s_net_layer == 0:
                curr_in_fc_sizes = [self.input_dim]
                curr_fc_sizes = self.fc_sizes_list
            elif s_net_layer == (self.num_s_net_layers - 1):
                curr_in_fc_sizes = self.fc_sizes_list
                curr_fc_sizes = [self.output_dim]
            else:
                curr_in_fc_sizes = list(self.fc_sizes_list) + [-1]
                curr_fc_sizes = self.fc_sizes_list

            for output_size_index, _output_size in enumerate(curr_fc_sizes):
                # String representing this op.
                curr_op_str = str(s_net_layer) + "," + str(output_size_index)

                # Index of mask values for this op.
                curr_param_ix = arch_params_indices[curr_op_str]

                # Mask values for this op.
                curr_mask_vals = sampled_mask_values[curr_param_ix]

                # Index of max mask value.
                o_size_max_ixs.append(int(np.argmax(curr_mask_vals)))

            # The FC size index that we want
            # to choose is output_size_max_
            # indices[previous_input_index] so
            # that the sizes of the ops and the
            # next layer's ops match. If we are
            # skipping this layer, the
            # previous_input_index should stay the same
            # because the next layer needs to preserve
            # the same output size.
            curr_fc_size_ix = o_size_max_ixs[previous_input_index]
            curr_chosen_fc_size = curr_in_fc_sizes[curr_fc_size_ix]

            # Do not add the layer or change the previous_input_index
            # if we are skipping the layer.
            if curr_chosen_fc_size != -1:
                prevoius_input_index = curr_fc_size_ix
                fc_sizes.append(curr_chosen_fc_size)

        # Return the list of FC sizes
        # in DLRM format.
        dlrm_format_fc_sizes = [self.input_dim] + \
                                fc_sizes + \
                                [self.output_dim]
        return dlrm_format_fc_sizes
