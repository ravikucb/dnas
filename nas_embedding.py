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


class EmbeddingDLRM(nn.Module):
    """
    Almost equivalent to nn.Embedding
    (some abilities of nn.Embedding do not exist in EmbeddingDLRM),
    except that it uses the weight initialization from
    dlrm_s_pytorch.py.
    """

    def __init__(self, num_features, embedding_dimension):
        # Superclass initialization.
        super(EmbeddingDLRM, self).__init__()

        # Store for later.
        self.num_features = num_features
        self.embedding_dimension = embedding_dimension

        # Create the embedding.
        self.embedding = nn.Embedding(num_features, embedding_dimension)

        with torch.no_grad():
            # Change embedding initialization to match
            # original DLRM code from dlrm_s_pytorch.py.
            W = np.random.uniform(low=-np.sqrt(1.0 / float(self.num_features)),
                                    high=np.sqrt(1.0 / float(self.num_features)),
                                    size=(self.num_features,
                                    self.embedding_dimension)).astype(np.float32)
            self.embedding.weight.data = torch.tensor(W, requires_grad=True)

    def forward(self, input_indices):
        """
        Run input_indices through the embedding.
        """

        return self.embedding(input_indices)
