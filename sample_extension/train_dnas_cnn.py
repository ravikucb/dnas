#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/23/21

# Various import statements.
import torch
import torch.nn as nn
from nas_searchmanager import SearchManager
from cnn_supernet import ConvSuperNet
from dnas_cnn_data_utils import CNNDataset
import argparse
import pickle
from utils import arch_sampling_str_to_dict, STR_TO_OPTIM
import random
import numpy as np
import hashlib
import time
import os

# Create argument parser.
parser = argparse.ArgumentParser(description="Run DNAS CNN test.")

# Training / search manager parameters.
parser.add_argument("--experiment_id",
                    type=str,
                    default=None,
                    help="Unique experiment ID used as a prefix for all files saved during the experiment, including sampled architectures and logfiles.")
parser.add_argument("--weights_batch_size",
                    type=int,
                    default=256,
                    help="Weights training batch size.")
parser.add_argument("--arch_params_batch_size",
                    type=int,
                    default=256,
                    help="Arch params training batch size.")
parser.add_argument("--initial_temperature",
                    type=float,
                    default=1.0,
                    help="Initial Gumbel Softmax temperature.")
parser.add_argument("--temp_decay_rate",
                    type=float,
                    default=0.1,
                    help="Decay rate of Gumbel Softmax temperature.")
parser.add_argument("--architecture_sampling",
                    type=str,
                    default="4:4",
                    help="Architecture sampling. To sample 4 architecture after 1 epoch of architecture parameters training, 4 after 2, etc. for all 4 epochs, one would write \"1:4,2:4,3:4,4:4\".")
parser.add_argument("--n_warmup_epochs",
                    type=float,
                    default=None,
                    help="Number (possibly float) of warmup epochs i.e. weights-only training before architecture parameters trained.")
parser.add_argument("--n_total_s_net_training_epochs",
                    type=float,
                    default=None,
                    help="Total (possibly float) number of supernet training epochs.")
parser.add_argument("--n_alt_train_epochs",
                    type=float,
                    default=1.0,
                    help="Every n_alt_train_epochs, we switch from training the weights to architecture parameters or vice versa.")
parser.add_argument("--host_gpu_id",
                    type=int,
                    default=None,
                    help="Host GPU ID.")
parser.add_argument("--clip_grad_norm_value",
                    type=float,
                    default=100.0,
                    help="L2 norm at which to clip gradients of supernet.")    # Both weights and architecture parameters gradients.
parser.add_argument("--weights_optim_type",
                    type=str,
                    choices=["sgd"],
                    default="sgd",
                    help="Weights optimizer type.")
parser.add_argument("--arch_params_optim_type",
                    type=str,
                    choices=["sgd", "adam", "adagrad"],
                    default="sgd",
                    help="Architecture parameters optimizer type.")
parser.add_argument("--weights_lr",
                    type=float,
                    default=None,
                    help="Initial learning rate for architecture weights.")
parser.add_argument("--arch_params_lr",
                    type=float,
                    default=None,
                    help="Initial learning rate for architecture configuration parameters.")
parser.add_argument("--weights_wd",
                    type=float,
                    default=0.0,
                    help="Weight decay for architecture weights.")
parser.add_argument("--arch_params_wd",
                    type=float,
                    default=0.0,
                    help="Weight decay for architecture configuration parameters.")
parser.add_argument("--use_hw_cost",
                    action="store_true",
                    help="Whether or not to use HW cost in the DNAS training.")
parser.add_argument("--hw_cost_function",
                    type=str,
                    choices=["exponential", "linear"],
                    default="linear",
                    help="HW cost function type if --use_hw_cost.")
parser.add_argument("--hw_cost_exp",
                    type=float,
                    default=None,
                    help="HW cost function exponent, provided only if --use_hw_cost and --hw_cost_function=exponential.")
parser.add_argument("--hw_cost_coef",
                    type=float,
                    default=0.001,
                    help="HW cost linear coefficient, provided if --use_hw_cost.")
parser.add_argument("--hw_cost_multiplier",
                    type=float,
                    default=1.0,
                    help="Linear HW cost multiplier to e.g. convert latency numbers measured in seconds to milliseconds.")
parser.add_argument("--weights_lr_base",
                    type=float,
                    default=0.9,
                    help="Weights LR = weights_lr * ((weights_lr_base) ** (num_weights_epochs)). Note that this formula may be applied at every training step or every n_alt_train_epochs.")    # Every epoch not currently an option - may be added later as an option.
parser.add_argument("--arch_params_lr_base",
                    type=float,
                    default=0.9,
                    help="Arch params LR = arch_params_lr * ((arch_params_lr_base) ** (num_arch_params_epochs)). Note that this formula may be applied at every training step or every n_alt_train_epochs.")    # Every epoch not currenly an option - may be added later.
parser.add_argument("--update_lrs_every_step",
                    action="store_true",
                    help="If set, LRs will be updated every step instead of every SearchManager \"epoch\" (usually args.n_alt_train_amt).")    # Could update the weights and architecture parameters learning rates at different frequencies.

# Seed.
parser.add_argument("--seed",
                    type=int,
                    default=1,
                    help="Random seed to ensure results can be replicated. This seed is used for random, numpy, and torch.")

# Needed to interface with tuning script.
parser.add_argument("--save_metrics_param",
                    type=str,
                    default="",
                    help="Path at which to save a file to tell tuning.py that this script is done running.")

# Parse arguments.
args = parser.parse_args()

# Set seed.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Create the supernet.
cnn_supernet = ConvSuperNet()

# Get dataloaders.
weights_dataloader, arch_params_dataloader = torch.utils.data.DataLoader(CNNDataset("train-weights"), batch_size=args.weights_batch_size), torch.utils.data.DataLoader(CNNDataset("train-archparams"), batch_size=args.arch_params_batch_size)

# Function to deal with OOM errors.
def write_oom_exit(oom_error):
    """
    Writes the text of the OOM error to the file
    and then exit()s.
    """

    # Write OOM error.
    oom_error_file = open(f"oom_error_{args.save_metrics_param}", "w")
    oom_error_file.write(str(oom_error))
    oom_error_file.flush()

    # Remove job information file.
    os.system(f"rm {job_info_filename}")

    # Exit.
    exit()

# Move DLRM supernet to GPU.
# Writing OOM error allows for job restarting.
try:
    host_device = torch.device(f"cuda:{args.host_gpu_id}")
    print(f"ATTEMPTING TO MOVE CNN SUPERNET TO GPU {args.host_gpu_id}.")
    print(cnn_supernet)
    cnn_supernet.to(host_device)
except RuntimeError as oom_error:
    write_oom_exit(oom_error)

# Construct various inputs to SearchManager.__init__():

# Optimizer classes.
weights_optim_class = STR_TO_OPTIM[args.weights_optim_type.lower()]
arch_params_optim_class = STR_TO_OPTIM[args.arch_params_optim_type.lower()]

# Optimizer initialization parameters.
weights_optim_init_params = {"lr" : args.weights_lr, "weight_decay" : args.weights_wd}
arch_params_optim_init_params = {"lr" : args.arch_params_lr, "weight_decay" : args.arch_params_wd}

# Functions to fetch parameters that each optimizer should train.
weights_parameters_function = lambda s_net: [param for param_name, param in s_net.named_parameters() if "theta_parameters" not in param_name]
arch_params_parameters_function = lambda s_net : [param for param_name, param in s_net.named_parameters() if "theta_parameters" in param_name]

# Functions which specify how the LR changes during training. Note that
# these functions return the RATIO of the current learning rate to the
# initial learning rate, and not the current learning rate itself.
weights_optim_lr_lambdas = [lambda curr_epoch: (args.weights_lr_base ** curr_epoch)]
arch_params_optim_lr_lambdas = [lambda curr_epoch: (args.arch_params_lr_base ** curr_epoch)]

# Initial learning rates for the different parameters groups in each
# optimizer. Currently there is only one parameter group used per optimizer,
# however, the code supports multiple parameters groups, each with their own
# initial learning rate and learning rate schedule.
weights_initial_lrs = [args.weights_lr]
arch_params_initial_lrs = [args.arch_params_lr]

# CrossEntropyLoss for image classification.
loss_function = nn.CrossEntropyLoss()

# Create search_manager.
search_manager = SearchManager(super_net=cnn_supernet,
                                init_temp=args.initial_temperature,
                                temp_decay_rate=args.temp_decay_rate,
                                n_warmup_epochs=args.n_warmup_epochs,
                                arch_sampling=arch_sampling_str_to_dict(args.architecture_sampling),
                                n_total_s_net_train_epochs=args.n_total_s_net_training_epochs,
                                n_alt_train_amt=args.n_alt_train_epochs,
                                host_device=host_device,
                                clip_grad_norm_value=args.clip_grad_norm_value,
                                w_dataloader=weights_dataloader,
                                m_dataloader=arch_params_dataloader,
                                w_optim_class=weights_optim_class,
                                weights_optim_init_params=weights_optim_init_params,
                                w_optim_params_func=weights_parameters_function,
                                m_optim_class=arch_params_optim_class,
                                mask_optim_init_params=arch_params_optim_init_params,
                                m_optim_params_func=arch_params_parameters_function,
                                weights_lr_lambdas=weights_optim_lr_lambdas,
                                mask_lr_lambdas=arch_params_optim_lr_lambdas,
                                weights_initial_lrs=weights_initial_lrs,
                                mask_initial_lrs=arch_params_initial_lrs,
                                update_lrs_every_step=args.update_lrs_every_step,
                                loss_function=loss_function,
                                experiment_id=args.experiment_id,
                                logfile=args.experiment_id.replace("save_file", "search_manager_logfile"),
                                use_hw_cost=args.use_hw_cost,
                                cost_exp=args.hw_cost_exp,
                                cost_coef=args.hw_cost_coef,
                                exponential_cost=(True if args.use_hw_cost and args.hw_cost_function == "exponential" else False),
                                cost_multiplier=args.hw_cost_multiplier)

# Start search process.
try:
    search_manager.train_dnas()
except RuntimeError as oom_error:
    write_oom_exit(oom_error)

# Once the DNAS process is done, in order to tuning.py to know
# that the script is done running, save a file at the save_metrics_param
# location.
with open(args.save_metrics_param, "wb") as save_metrics_writefile:
    pickle.dump({"info" : "SCRIPT COMPLETED"}, save_metrics_writefile)
