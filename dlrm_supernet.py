#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Ravi Krishna 07/25/21

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dlrm_s_pytorch import DLRM_Net
from nas_supernet import SuperNet
from nas_mlp import SuperNetMLP
from nas_embedding_dim import EmbeddingDimSuperNet
from nas_embedding_card import EmbeddingCardSuperNet

class DLRMSuperNet(DLRM_Net, SuperNet):
    """
    Defines a DNAS supernet structure with the Deep Learning Recommendation
    Model (DLRM) backbone network. Currently implements MLP search via
    FC of FC by setting the bottom and top MLPs to be SuperNetMLP objects.
    """

    def gen_hw_cost_table(self,
                        bottom_mlp_sizes,
                        top_mlp_sizes):
        """
        Generates a HW cost table which contains
        the number of FLOPs used by each FC operator.

        Operators are defined by their input and
        output dimensions.
        """

        # Get list of input and output sizes.
        fc_sizes_list = list(np.unique(bottom_mlp_sizes + \
                                        top_mlp_sizes + \
                                        [self.n_dense_features] + \
                                        [self.top_mlp_input_dim] + \
                                        [self.emb_dim] + \
                                        [1]))

        # Create indices look up table and cost table.
        indices_table = {}
        cost_table = []

        # Go through each possible operator. Size of -1 indicate the option to skip a layer.
        curr_op_ix = 0
        for input_size in (fc_sizes_list + [-1]):
            for output_size in fc_sizes_list:
                # Operator name.
                curr_op_name = f"fc_{input_size}_{output_size}"

                # If this operator is an operator to allow skipping layers, the cost should be 0.
                if (input_size == -1) or (output_size == -1):
                    total_op_flops = 0
                else:
                    # Calculate the number of FLOPs.
                    weight_flops = output_size * ((2 * input_size) - 1)
                    bias_flops = output_size
                    total_op_flops = weight_flops + bias_flops

                # Create entries.
                indices_table[curr_op_name] = curr_op_ix
                cost_table.append(float(total_op_flops))

                # Increment operator index.
                curr_op_ix += 1

        # Convert the cost table to a PyTorch tensor.
        cost_table = torch.tensor(cost_table)

        # Return the indices look up table and cost table.
        return indices_table, cost_table

    def __init__(self,
                search_space,
                emb_dim,
                embs_n_vectors,
                n_dense_features,
                bottom_mlp_sizes,
                max_n_bottom_mlp_layers,
                top_mlp_sizes,
                max_n_top_mlp_layers,
                interaction_op="dot",
                include_self_interaction=False,
                gen_cost_table=True,
                indices_look_up_table=None,
                cost_table=None,
                loss_threshold=1e-9,
                emb_card_options=None,
                enable_float_card_options=False):

        """
        Initializes the DLRMSuperNet by first initializing a regular DLRM_Net
        model, and then replacing the bottom and top MLPs with SuperNetMLP
        instances. After that we add the other miscellaneous attributes that
        SuperNet requires.
        """

        # Store for later.
        self.search_space = search_space
        self.emb_dim = emb_dim
        self.embs_n_vectors = embs_n_vectors
        self.n_dense_features = n_dense_features
        self.bottom_mlp_sizes = bottom_mlp_sizes
        self.max_n_bottom_mlp_layers = max_n_bottom_mlp_layers
        self.top_mlp_sizes = top_mlp_sizes
        self.max_n_top_mlp_layers = max_n_top_mlp_layers
        self.interaction_op = interaction_op
        self.include_self_interaction = include_self_interaction
        self.gen_cost_table = gen_cost_table
        self.loss_threshold = loss_threshold
        self.emb_card_options = emb_card_options
        self.enable_float_card_options = enable_float_card_options

        # Embedding dimension.
        self.actual_emb_dim = (max(self.emb_dim) if self.search_space == "emb_dim" else self.emb_dim)

        # Calculate top MLP input dimension,
        # used to initialize DLRM_Net.
        num_features = int(1 + len(self.embs_n_vectors))
        if self.interaction_op == "dot":
            num_feature_interactions = int((num_features * (num_features - 1)) / 2)
            self.top_mlp_input_dim = num_feature_interactions + self.actual_emb_dim
        elif self.interaction_op == "cat":
            self.top_mlp_input_dim = num_features * self.actual_emb_dim

        # If needed, generate the cost table.
        if not self.gen_cost_table:
            self.indices_look_up_table = indices_look_up_table
            self.cost_table = torch.tensor(cost_table)
        else:
            if self.search_space == "top_bottom_mlps":
                self.indices_look_up_table, self.cost_table = \
                    self.gen_hw_cost_table(self.bottom_mlp_sizes,
                                            self.top_mlp_sizes)
            else:
                self.indices_look_up_table, self.cost_table = None, None

        # Create bottom and top MLP structures
        # in order to initialize the DLRM_Net;
        # these are never actually used so they
        # do not matter.
        dlrm_ln_bot = np.array(self.bottom_mlp_sizes) if self.search_space in ["emb_dim", "emb_card"] else np.array([self.n_dense_features, self.actual_emb_dim])
        dlrm_ln_top = np.array(self.top_mlp_sizes) if self.search_space in ["emb_dim", "emb_card"] else np.array([self.top_mlp_input_dim, 1])

        # Initialize the DLRM_Net model. If emb dimension corresponds to a search space then
        # pass the maximum dimension.
        DLRM_Net.__init__(self,
                            m_spa=self.actual_emb_dim,
                            ln_emb=np.array(self.embs_n_vectors),
                            ln_bot=dlrm_ln_bot,
                            ln_top=dlrm_ln_top,
                            arch_interaction_op=self.interaction_op,
                            arch_interaction_itself=self.include_self_interaction,
                            sigmoid_top=((len(dlrm_ln_top) - 2) if self.search_space in ["emb_dim", "emb_card"] else 1),
                            loss_threshold=loss_threshold)

        # Now that we have initialized DLRM_Net,
        # replace the bottom the top MLPs,
        # and create the other parameters that
        # SuperNet requires.

        # Depending on search space, replace pieces of DLRM with
        # different supernets.
        if self.search_space == "top_bottom_mlps":
            # Replace the bottom MLP.
            self.bot_l = SuperNetMLP(input_dim=self.n_dense_features,
                                    output_dim=self.actual_emb_dim,
                                    num_s_net_layers=self.max_n_bottom_mlp_layers,
                                    fc_sizes_list=self.bottom_mlp_sizes,
                                    indices_look_up_table=self.indices_look_up_table,
                                    cost_table=self.cost_table,
                                    relu_last_layer=True,
                                    bias=True,
                                    last_layer_bias=True,
                                    last_layer_activation=F.relu)

            # Replace the top MLP.
            self.top_l = SuperNetMLP(input_dim=self.top_mlp_input_dim,
                                    output_dim=1,
                                    num_s_net_layers=self.max_n_top_mlp_layers,
                                    fc_sizes_list=self.top_mlp_sizes,
                                    indices_look_up_table=self.indices_look_up_table,
                                    cost_table=self.cost_table,
                                    relu_last_layer=True,
                                    bias=True,
                                    last_layer_bias=True,
                                    last_layer_activation=torch.sigmoid)

            # Set the theta parameters, mask values, and the number of mask values.

            # Theta parameters.
            self.theta_parameters = nn.ParameterList([])
            for bottom_parameter in self.bot_l.theta_parameters:
                self.theta_parameters.append(bottom_parameter)
            for top_parameter in self.top_l.theta_parameters:
                self.theta_parameters.append(top_parameter)

            # Mask values.
            self.mask_values = self.bot_l.mask_values + self.top_l.mask_values

            # Numbers of mask lists.
            self.num_bottom_mask_values = len(self.bot_l.theta_parameters)
            self.num_top_mask_values = len(self.top_l.theta_parameters)
            self.num_mask_lists = self.num_bottom_mask_values + self.num_top_mask_values

        elif self.search_space == "emb_dim":
            # Replace each embedding.
            for emb_ix in range(len(self.embs_n_vectors)):
                self.emb_l[emb_ix] = EmbeddingDimSuperNet(self.embs_n_vectors[emb_ix], self.emb_dim)

            # Set the theta parameters, mask values, and the number of mask values.

            # Theta parameters.
            self.theta_parameters = nn.ParameterList([])
            for emb_ix in range(len(self.embs_n_vectors)):
                 self.theta_parameters.append(self.emb_l[emb_ix].theta_parameters[0])

            # Mask values.
            self.mask_values = []
            for emb_ix in range(len(self.embs_n_vectors)): self.mask_values += self.emb_l[emb_ix].mask_values

            # Number of mask lists.
            self.num_mask_lists = len(self.mask_values)

        elif self.search_space == "emb_card":
            # Replace each embedding.
            for emb_ix in range(len(self.embs_n_vectors)):
                # Current cardinality options are either directly specific integer cardinalities
                # or proportion-determined cardinalities using original max cardinalities.
                if not self.enable_float_card_options:
                    curr_card_options = self.emb_card_options
                else:
                    curr_card_options = [max(int(card_option), 1) for card_option in np.array(self.emb_card_options) * self.embs_n_vectors[emb_ix]]
                    print(f"Cardinality options calculated as {curr_card_options} with float options {self.emb_card_options} and current max cardinality {self.embs_n_vectors[emb_ix]}")

                self.emb_l[emb_ix] = EmbeddingCardSuperNet(curr_card_options, self.emb_dim)

            # Set the theta parameters, mask values, and number of mask values.

            # Theta parameters.
            self.theta_parameters = nn.ParameterList([])
            for emb_ix in range(len(self.embs_n_vectors)):
                self.theta_parameters.append(self.emb_l[emb_ix].theta_parameters[0])

            # Mask values.
            self.mask_values = []
            for emb_ix in range(len(self.embs_n_vectors)): self.mask_values += self.emb_l[emb_ix].mask_values

            # Number of mask lists.
            self.num_mask_lists = len(self.mask_values)

        else:
            raise ValueError(f"{self.search_space} IS NOT A CURRENTLY SUPPORTED DLRMSuperNet SEARCH SPACE.")

    def to(self, device):
        """
        Regular to() function does not move
        the bottom and top MLP cost tables
        to the device. This one does so
        directly. There is probably a better
        way to fix this - the cost table
        is probably not registering as a
        parameter properly.
        """

        # Call the original function.
        nn.Module.to(self, device)

        # Move the cost tables if needed.
        if self.search_space == "top_bottom_mlps":
            self.bot_l.cost_table = self.bot_l.cost_table.to(device)
            self.top_l.cost_table = self.top_l.cost_table.to(device)

    def forward(self,
                dense_features,
                sparse_offsets,
                sparse_indices,
                sampling="None",
                temperature=-1.0):

        """
        Runs the forward pass for the DLRM supernet.
        """

        # Get the batch size of the input.
        input_batch_size = int(list(dense_features.size())[0])

        # Run sampling if necessary. WILL THIS SAMPLE TWICE?
        if sampling == "soft":
            print("RUNNING SOFT SAMPLING.")
            if self.search_space == "top_bottom_mlps":
                self.bot_l.soft_sample(temperature, input_batch_size)
                self.top_l.soft_sample(temperature, input_batch_size)

                self.mask_values = self.bot_l.mask_values + \
                    self.top_l.mask_values

            elif self.search_space in ["emb_dim", "emb_card"]:
                # Run sampling and store mask values.
                self.mask_values = []
                for emb_ix in range(len(self.embs_n_vectors)):
                    self.emb_l[emb_ix].soft_sample(temperature, input_batch_size)
                    self.mask_values += self.emb_l[emb_ix].mask_values

        # Call DLRM_Net.forward(); this will also
        # set self.bot_l.curr_cost and
        # self.top_l.curr_cost.
        click_probabilities = DLRM_Net.forward(self,
                                                dense_x=dense_features,
                                                lS_o=sparse_offsets,
                                                lS_i=sparse_indices)

        # Calculate the total cost from the bottom
        # and top MLPs.
        if self.search_space == "top_bottom_mlps":
            total_cost = self.bot_l.curr_cost + \
                        self.top_l.curr_cost

        elif self.search_space in ["emb_dim", "emb_card"]:
            total_cost = sum([self.emb_l[emb_ix].curr_cost for emb_ix in range(len(self.embs_n_vectors))])

        # Return the click probabilities and
        # the total cost of the MLPs.
        return click_probabilities, total_cost

    def sample_arch(self):
        """
        Samples from the bottom and top MLP
        supernets by calling
        self.bot_l.sample_mlp_arch() as well
        as self.top_l.sample_mlp_arch().

        Returns arguments necessary to create
        a DLRM_Net with the sampled MLP
        architectures.
        """

        if self.search_space == "top_bottom_mlps":
            # Sample the bottom MLP architecture.
            bottom_mlp_arch = self.bot_l.sample_mlp_arch()

            # Sample the top MLP architecture.
            top_mlp_arch = self.top_l.sample_mlp_arch()

            # Number of top MLP layers; used to
            # specify layer at which to use
            # sigmoid activation.
            num_top_mlp_layers = len(top_mlp_arch) - 1

            # Arguments to initialize sampled DLRM_Net.
            sampled_dlrm_args = {"m_spa" : self.emb_dim,
                                "ln_emb" : np.array(self.embs_n_vectors),
                                "ln_bot" : np.array(bottom_mlp_arch),
                                "ln_top" : np.array(top_mlp_arch),
                                "arch_interaction_op" : self.interaction_op,
                                "arch_interaction_itself" : self.include_self_interaction,
                                "sigmoid_top" : (num_top_mlp_layers - 1),
                                "loss_threshold" : self.loss_threshold}

        elif self.search_space == "emb_dim":
            # Sample the embedding architectures.
            sampled_embedding_dimensions = [self.emb_l[emb_ix].sample_emb_arch() for emb_ix in range(len(self.embs_n_vectors))]

            # Get the maximum embedding dimension.
            max_emb_dim = max(sampled_embedding_dimensions)

            # Arguments to initialize sampled DLRM_Net.
            sampled_dlrm_args = {"m_spa" : sampled_embedding_dimensions,
                                "ln_emb" : np.array(self.embs_n_vectors),
                                "ln_bot" : np.array(self.bottom_mlp_sizes),
                                "ln_top" : np.array(self.top_mlp_sizes),
                                "arch_interaction_op" : self.interaction_op,
                                "arch_interaction_itself" : self.include_self_interaction,
                                "sigmoid_top" : (len(self.top_mlp_sizes) - 2),
                                "loss_threshold" : self.loss_threshold,
                                "max_emb_dim" : max_emb_dim}

        elif self.search_space == "emb_card":
            # Sample the embedding architectures.
            sampled_embedding_cards = [self.emb_l[emb_ix].sample_emb_arch() for emb_ix in range(len(self.embs_n_vectors))]

            # Arcuments to initialize sampled DLRM_Net.
            sampled_dlrm_args = {"m_spa" : self.emb_dim,
                                 "ln_emb" : np.array(sampled_embedding_cards),
                                 "ln_bot" : np.array(self.bottom_mlp_sizes),
                                 "ln_top" : np.array(self.top_mlp_sizes),
                                 "arch_interaction_op" : self.interaction_op,
                                 "arch_interaction_itself" : self.include_self_interaction,
                                 "sigmoid_top" : (len(self.top_mlp_sizes) - 2),
                                 "loss_threshold" : self.loss_threshold}

        # Return the sampled DLRM arguments.
        return sampled_dlrm_args
