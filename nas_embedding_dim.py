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
from nas_embedding import EmbeddingDLRM
from nas_supernet import SuperNet
import numpy as np

class EmbeddingDimSuperNet(SuperNet):
    def __init__(self,
                cardinality,
                dim_options):
        """
        Implements an embedding dimension search supernet.

        Adopts an FBNetv2 approach to searching over the number of
        channels in an embedding by storing a single embedding
        matrix of the maximum possible dimension and then taking
        a weighted sum over truncated and then zero-padded versions
        of the maximum dimension version of an embedding vector.
        """

        # Superclass initialization.
        super(EmbeddingDimSuperNet, self).__init__()

        # Store for later.
        self.cardinality = cardinality
        self.dim_options = dim_options
        self.max_dim = max(dim_options)
        self.num_dim_options = len(self.dim_options)
        self.params_options = nn.Parameter(torch.Tensor([curr_dim * self.cardinality for curr_dim in self.dim_options]), requires_grad=False)

        # Necessary for compatibility with modification to dlrm_s_pytorch.py for embedding cardinality supernet.
        self.num_embeddings = self.cardinality

        # Create largest dim embedding matrix.
        self.largest_embedding = EmbeddingDLRM(self.cardinality, self.max_dim)

        # Create other parameters.
        self.theta_parameters = nn.ParameterList([nn.Parameter(torch.Tensor([0.00] * self.num_dim_options), requires_grad=True)])
        self.mask_values = [None] * len(self.theta_parameters)
        self.num_mask_lists = len(self.mask_values)

        # For compatibility with DLRM, store the current cost
        # instead of returning it after each call to forward().
        self.curr_cost = None

    def calculate_cost(self):
        """
        Calculates the cost as the weighted average number of parameters.
        """

        # Get the mask values. This will be a tensor of size
        # (batch_size, len(self.dim_options)).
        curr_mask_values = self.mask_values[0]

        # Take the dot product with the number of parameters for each
        # of the dimension options. Should be of size (batch_size).
        #print(f"CURRENT MASK VALUES = {curr_mask_values.size()}, PARAMS OPTIONS = {self.params_options.size()}")
        weighted_avg_cost = torch.matmul(curr_mask_values, self.params_options)

        # Return weighted average cost.
        return weighted_avg_cost

    def to(self, device):
        """
        Overrides the original to() and also moves self.params_options
        and self.theta_parameters.
        """

        nn.Module.to(self, device)
        self.params_options = self.params_options.to(device)
        self.theta_parameters = self.theta_parameters.to(device)

    def forward(self, indices, offsets, sampling="None", temperature=-1.0):
        """
        Note that DLRM actually uses an nn.EmbeddingBag, not an nn.Embedding.

        Thus, the input includes both indices and offsets. However, this
        SuperNet only implements an nn.Embedding search and ignores these offsets.
        """

        # Get the batch size.
        curr_batch_size = int(list(indices.size())[0])

        # Run the sampling if necessary.
        if sampling == "soft":
            self.soft_sample(temperature, curr_batch_size)

        # Calculate the cost of the network.
        self.curr_cost = self.calculate_cost()

	# Get the largest dimension output.
        largest_dim_output = self.largest_embedding(indices)

        # Create truncated, zero-padded outputs.
        dim_outputs = []
        for current_dim in self.dim_options:
            # Truncate original output.
            curr_truncated_output = largest_dim_output[:, : current_dim]

            # Zero-pad the truncated output.
            curr_padded_truncated_output = F.pad(curr_truncated_output, (0, (self.max_dim - current_dim)))

            # Add to list of outputs.
            dim_outputs.append(curr_padded_truncated_output)

        # Take the weighted average of the outputs.
        weighted_average_output = self.calculate_weighted_sum(self.mask_values[0], dim_outputs, n_mats=self.num_dim_options)

        # Return the weighted average output.
        return weighted_average_output

    def sample_emb_arch(self):
        """
        Hard-samples from the Gumbel Softmax distribution
        and returns the resulting embedding dimension as
        an integer.
        """

	# Hard-sample from the distribution.
        sampled_mask_values = self.hard_sample()

        # Find the one-hot index of the mask.
        one_hot_ix = np.argmax(sampled_mask_values[0])

        # Return the embedding dimension size at
        # that index.
        sampled_size = self.dim_options[one_hot_ix]
        return sampled_size

if __name__ == "__main__":
    # Create a supernet.
    supernet = EmbeddingDimSuperNet(10000, [8, 16, 32, 64])

    # Move it to GPU.
    supernet.to("cuda:4")
    print(f"PARAMS OPTIONS DEVICE: {supernet.params_options.device}")
    print(f"THETA PAREMETERS DEVICE: {supernet.theta_parameters[0].device}")

    # Run the forward pass with a batch size of 10.
    random_indices = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to(dtype=torch.long).to("cuda:4")
    random_offsets = None
    sampling_type = "soft"
    temperature = 0.1

    avg_embs = supernet(indices=random_indices, offsets=random_offsets, sampling=sampling_type, temperature=temperature)
    print(f"SIZE OF AVERAGE EMBEDDINGS TENSOR: {avg_embs.size()}")
    print(f"MASK VALUES: {supernet.mask_values[0]}")
    print(f"CURRENT COST: {supernet.curr_cost}")

    # Try to sample architectures.
    print(f"CURRENT THETA PARAMETERS: {supernet.theta_parameters[0]}")
    archs = {k : 0 for k in [8, 16, 32, 64]}
    for i in range(10000):
        with torch.no_grad():
            curr_arch = supernet.sample_emb_arch()
            archs[curr_arch] += 1
            print(f"SAMPLED ARCHITECTURE: {curr_arch} {archs}")
