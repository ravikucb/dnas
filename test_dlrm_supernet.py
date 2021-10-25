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
import dlrm_data_pytorch as dp
import sys
import pickle
import random
import os
from dlrm_supernet import DLRMSuperNet

# CONSTANTS
search_space = sys.argv[1]
enable_float_options = (sys.argv[2] == "1")
emb_dim = 8 if sys.argv[1] == "top_bottom_mlps" or sys.argv[1] == "emb_card" else [8, 16, 32, 64]
embs_n_vectors = [1000, 10000, 100000]
n_dense_features = 10
bottom_mlp_sizes = [8, 16, 32] if sys.argv[1] == "top_bottom_mlps" else ([10, 16, 16, 8] if sys.argv[1] == "emb_card" else [10, 16, 16, 64])
max_n_bottom_mlp_layers = 8
top_mlp_sizes = [16, 32, 64] if sys.argv[1] == "top_bottom_mlps" else ([6 + 8, 16, 16, 1] if sys.argv[1] == "emb_card" else [6 + 64, 16, 16, 1])
max_n_top_mlp_layers = 8
interaction_op = "dot"
include_self_interaction = False
gen_cost_table = True
indices_look_up_table = None
cost_table = None
loss_threshold = 1e-9
emb_card_options = ([1000, 100, 10] if not enable_float_options else [1.0, 0.1, 0.01]) if sys.argv[1] == "emb_card" else None

weights_lr = 0.001
mask_lr = 0.001

gpu_id = int(sys.argv[3])

# Sample a large number of architcetures to
# confirm that allowing a variable number of
# layers does not cause problems.
num_architectures_to_sample = 500

# This function is taken directly from the Facebook
# TBSM repo's tbsm_pytorch.py.
def reset_seed(seed, use_gpu=True):
    """
    Resets seed to allow for comparing outputs directly.

    Code taken directly from tbsm_pytorch.py from Facebook TBSM repo.
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if use_gpu:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # if using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class DataArgs(object):
    """
    Holds the arguments that would normally
    be collected by argparse.
    """

    def __init__(self):
        self.data_size = 10000
        self.num_batches = 20
        self.mini_batch_size = 8
        self.num_indices_per_lookup = 1
        self.num_indices_per_lookup_fixed = True
        self.round_targets = False
        self.data_generation = "random"
        self.data_trace_file = ""
        self.data_trace_enable_padding = False
        self.numpy_rand_seed = 1
        self.num_workers = 0

# Very slightly modified from dlrm_s_pytorch.py
def move_data_to_gpu(X, lS_o, lS_i, T, device):
    # lS_i can be either a list of tensors or a stacked tensor.
    # Handle each case below:
    lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
        else lS_i.to(device)
    lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
        else lS_o.to(device)

    return X.to(device), \
            lS_o, \
            lS_i, \
            T.to(device)


def create_test_super_net():
    """
    Tests whether or not we can create
    the supernet at all.
    """

    # Create an example DLRMSuperNet.
    test_s_net = DLRMSuperNet(search_space=search_space,
                    emb_dim=emb_dim,
                    embs_n_vectors=embs_n_vectors,
                    n_dense_features=n_dense_features,
                    bottom_mlp_sizes=bottom_mlp_sizes,
                    max_n_bottom_mlp_layers=max_n_bottom_mlp_layers,
                    top_mlp_sizes=top_mlp_sizes,
                    max_n_top_mlp_layers=max_n_top_mlp_layers,
                    interaction_op=interaction_op,
                    include_self_interaction=include_self_interaction,
                    gen_cost_table=gen_cost_table,
                    indices_look_up_table=indices_look_up_table,
                    cost_table=cost_table,
                    loss_threshold=loss_threshold,
                    emb_card_options=emb_card_options,
                    enable_float_card_options=enable_float_options)

    # Print the net.
    print(test_s_net)

    # Return the net.
    return test_s_net

def run_super_net_fwd_on_gpu(gpu_id):
    """
    Creates super net, moves it to the
    GPU with ID gpu_id, and runs the
    forward pass with random data.
    """

    # Create the GPU device.
    gpu_device = torch.device(f"cuda:{gpu_id}")

    # Create the super net.
    s_net = create_test_super_net()

    # Move the super net to GPU.
    s_net.to(gpu_device)

    # Run example data through the network.
    # The following lines to create the
    # dataloader are taken directly
    # from dlrm_s_pytorch.py.

    ln_emb = embs_n_vectors
    m_den = n_dense_features
    train_data, train_ld = dp.make_random_data_and_loader(DataArgs(), ln_emb, m_den)

    # Run through the forward pass loop.
    for batch_ix, (dense_features, sparse_offsets, sparse_indices, labels) in enumerate(train_ld):
        # Move the data to the GPU device.
        dense_features, sparse_offsets, sparse_indices, labels = \
                move_data_to_gpu(dense_features, sparse_offsets, sparse_indices, labels, gpu_device)

        # Run the forward pass.
        dlrm_s_net_output, dlrm_s_net_cost = s_net(dense_features, sparse_offsets, sparse_indices, sampling="soft", temperature=1.0)

        # Print the output.
        print(f"BATCH {batch_ix}, SUPERNET OUTPUT: {dlrm_s_net_output}, SUPERNET COST: {dlrm_s_net_cost}")

    # Return the model.
    return s_net

def run_super_net_fwd_bckwd_on_gpu(gpu_id, to_optimize="weights,mask"):
    """
    Same as run_super_net_fwd_on_gpu, except
    that it also runs the backward pass with
    separate weights and architecture parameter
    optimizers.
    """

    # Create the GPU device.
    gpu_device = torch.device(f"cuda:{gpu_id}")

    # Create the super net.
    s_net = create_test_super_net()

    # Move the super net to GPU.
    s_net.to(gpu_device)

    # Create weights optimizer.
    weights_optim = torch.optim.SGD(list(s_net.bot_l.parameters()) + list(s_net.top_l.parameters()) + list(s_net.emb_l.parameters()), lr=weights_lr)

    # Create mask optimizer.
    mask_optim = torch.optim.SGD(s_net.theta_parameters.parameters(), lr=mask_lr)

    # Create the loss function.
    loss = nn.BCELoss()

    # Run example data through the network.
    # The following lines to create the
    # dataloader are taken directly
    # from dlrm_s_pytorch.py.

    ln_emb = embs_n_vectors
    m_den = n_dense_features
    train_data, train_ld = dp.make_random_data_and_loader(DataArgs(), ln_emb, m_den)

    # Run through the forward pass loop.
    for batch_ix, (dense_features, sparse_offsets, sparse_indices, labels) in enumerate(train_ld):
        # Zero gradients.
        weights_optim.zero_grad()
        mask_optim.zero_grad()

        # Move the data to the GPU device.
        dense_features, sparse_offsets, sparse_indices, labels = \
                move_data_to_gpu(dense_features, sparse_offsets, sparse_indices, labels, gpu_device)

        # Run the forward pass.
        dlrm_s_net_output, dlrm_s_net_cost = s_net(dense_features, sparse_offsets, sparse_indices, sampling="soft", temperature=1.0)

        # Calculate the loss.
        curr_loss = loss(dlrm_s_net_output, labels)

        # Print the output.
        print(f"BATCH {batch_ix}, SUPERNET OUTPUT: {dlrm_s_net_output}, SUPERNET COST: {dlrm_s_net_cost}, BCE LOSS: {curr_loss.item()}")

        # Backward.
        curr_loss.backward()

        if "weights" in to_optimize:
            # Update the weights.
            weights_optim.step()

        if "mask" in to_optimize:
            # Update the mask.
            mask_optim.step()

    # Return the model.
    return s_net

def run_super_net_fwd_bckwd_on_gpu_and_sample(gpu_id, num_architectures_to_sample):
    """
    Same as run_super_net_fwd_bckwd_on_gpu,
    except that is also samples multiple
    architectures and confirms that DLRM_Net
    instances can be created based on them.

    Because the code works directly with DLRM_Net
    which is already extensively used, further
    testing to confirm the functionality for the
    forward and backward of DLRM_Net should not
    be needed.
    """

    # Train both weights and mask on GPU.
    s_net = run_super_net_fwd_bckwd_on_gpu(gpu_id, to_optimize="weights,mask")

    # Sample architectures.
    sampled_architectures = []
    for curr_arch_ix in range(num_architectures_to_sample):
        # Sample an architecture.
        curr_sampled_arch = s_net.sample_arch()

        # Save the architecture configuration.
        sampled_architectures.append(curr_sampled_arch)

        # Print sampled architecture.
        print(f"ARCHITECTURE {curr_arch_ix}, CONFIGURATION {curr_sampled_arch}")

        # Create a DLRM_Net based
        # on this architecture.
        curr_dlrm = DLRM_Net(**curr_sampled_arch)
        print("DLRM_Net successfully created with this architecture!")

    # Return the sampled architectures.
    return sampled_architectures

# Set seed to 1.
reset_seed(1)

# Run all of the tests.
passed_all = True
try:
    s_net = create_test_super_net()
    print("PASSED create_test_super_net TEST")
except Exception as e:
    print(f"FAILED create_test_super_net TEST: {e}")
    passed_all = False

try:
    s_net = run_super_net_fwd_on_gpu(gpu_id)
    print("PASSED run_super_net_fwd_on_gpu TEST")
except Exception as e:
    print(f"FAILED run_super_net_fwd_on_gpu TEST: {e}")
    passed_all = False

try:
    s_net = run_super_net_fwd_bckwd_on_gpu(gpu_id, "weights")
    s_net = run_super_net_fwd_bckwd_on_gpu(gpu_id, "mask")
    s_net = run_super_net_fwd_bckwd_on_gpu(gpu_id, "weights,mask")
    print("PASSED run_super_net_fwd_bckwd_on_gpu TEST")
except Exception as e:
    print(f"FAILED run_super_net_fwd_bckwd_on_gpu TEST: {e}")
    passed_all = False

try:
    sampled_architectures = run_super_net_fwd_bckwd_on_gpu_and_sample(gpu_id, num_architectures_to_sample)
    print("PASSED run_super_net_fwd_bckwd_on_gpu_and_sample TEST")
except Exception as e:
    print(f"FAILED run_super_net_fwd_bckwd_on_gpu_and_sample TEST: {e}")
    passed_all = False

print(f"PASSED ALL TESTS: {passed_all}")
