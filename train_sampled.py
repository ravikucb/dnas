#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Ravi Krishna 10/07/20

# Various import statements.
import torch
import torch.nn as nn
from dlrm_s_pytorch import DLRM_Net
import dlrm_data_pytorch as dp
import argparse
from dnas_data_utils import get_dnas_dataloaders
import pickle
from utils import STR_TO_OPTIM
import random
import numpy as np
from torch.nn.utils import clip_grad_norm_ as clip_grad
from utils import step_lambda_lr
import time
import os

# Utility function.
# Very slightly modified from dlrm_s_pytorch.py.
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

# Evaluation function.
def evaluate(dlrm, evaluation_dataloader):
    """
    Evaluates dlrm on evaluation_dataloader dataset. This can
    be either the validation of test dataset.

    Returns BCE loss and calibration on the dataset.
    """

    # In order for model training results to be the same
    # regardless of whether we evaluate, we need to save the
    # CPU and CUDA RNG states before evaluation and load them
    # back after evaluation.
    original_cpu_rng_state = torch.get_rng_state()
    original_gpu_rng_state = torch.cuda.get_rng_state()

    # Create loss function.
    loss_function = nn.BCELoss()

    # Store the total loss and number of data points.
    total_loss = 0.0
    num_data_points = 0

    # Store the total sum of predictions and labels
    # to calculate calibration.
    predictions_sum = 0.0
    labels_sum = 0.0

    for iter, (X, lS_o, lS_i, T) in enumerate(evaluation_dataloader):
        # Move data to GPU.
        dense_features, sparse_offsets, sparse_indices, labels = \
                move_data_to_gpu(X, lS_o, lS_i, T, host_device)

        # Forward pass.
        click_probabilities = dlrm(dense_features, sparse_offsets, sparse_indices)

        # Calcuate loss.
        loss = loss_function(click_probabilities, labels)

        # Add to total loss, update total data points.
        curr_batch_size = list(labels.size())[0]
        total_loss += (loss.item()) * curr_batch_size
        num_data_points += curr_batch_size

        # Update predictions and labels sum.
        predictions_sum += float(torch.sum(click_probabilities))
        labels_sum += float(torch.sum(labels))

    # Calculate overall loss and calibration.
    dataset_loss = total_loss / num_data_points
    dataset_calibration = predictions_sum / labels_sum

    # Re-load original RNG states.
    torch.set_rng_state(original_cpu_rng_state)
    torch.cuda.set_rng_state(original_gpu_rng_state)

    # Return results.
    return {"loss" : dataset_loss, "calibration" : dataset_calibration}

# Create argument parser.
parser = argparse.ArgumentParser(description="Run DNAS DLRM supernet training.")

parser.add_argument("--saved_arch_filename",
                    type=str,
                    default="",
                    help="Saved architecture path.")
parser.add_argument("--tuning_experiment_id",
                    type=str,
                    default="",
                    help="Tuning experiment ID used for saving results files.")
parser.add_argument("--embeddings_num_vectors",
                    type=int,
                    nargs="+",
                    help="Number of vectors for each embedding table.")
parser.add_argument("--feature_counts_file",
                    type=str,
                    default="",
                    help="Feature counts file for emb_card search space.")

# Dataset parameters.
parser.add_argument("--dataset",
                    type=str,
                    choices=["kaggle"],
                    default="kaggle",
                    help="Dataset to use; currently only kaggle supported.")
parser.add_argument("--memory_map",
                    action="store_true",
                    help="Split up dataset across days.")
parser.add_argument("--no_click_subsample_rate",
                    type=float,
                    default=0.0,
                    help="Subsampling rate of no-click data points; DLRM repo uses 0.875 for Terabyte dataset.")
parser.add_argument("--overall_subsample_rate",
                    type=float,
                    default=0.0,
                    help="Subsample rate for entire dataset. Applied after no click subsampling. 0.0 = remove 0% of data points, 1.0 = remove 100% of data points.")
parser.add_argument("--raw_data_file",
                    type=str,
                    default="",
                    help="Raw dataset file.")
parser.add_argument("--processed_data_file",
                    type=str,
                    default="",
                    help="Pre-processed data file input.")
parser.add_argument("--load_processed",
                    action="store_true",
                    help="Code loads from processed data if true.")
parser.add_argument("--dataset-multiprocessing",
                    action="store_true",
                    help="Whether or not to use DLRM dataset multiprocessing.")
parser.add_argument("--data_randomization",
                    type=str,
                    choices=["total", "day", "none"],
                    default="total",
                    help="Data randomization method.")
parser.add_argument("--category_dropping",
                    type=str,
                    choices=["none", "modulo", "frequency"],
                    default="frequency",
                    help="Method by which categories in categorical features are dropped. If \"none\" is selected, all categorical indices are kept the same. \"modulo\" will hash the indices of each feature to the number of vectors specified in --embeddings_num_vectors, and \"frequency\" will drop the rarest categories for each features, mapping them to the missing category, such that the number of remaining categories matches the number specified in --embeddings_num_vectors. Note that if \"modulo\" is specified, the same number of vectors should be allowed for each feature so that the max_ind_range argument to getCriteoAdData can be used.")
parser.add_argument("--use_random_sampler_train_dataset",
                    action="store_true",
                    help="Whether or not to use a RandomSampler for the training dataset. This will have the effect of completely randomizing the ordering of the data, as well as what data is sampled every epoch.")
parser.add_argument("--use_random_sampler_val_dataset",
                    action="store_true",
                    help="Whether or not to use a RandomSampler for the validation dataset. This will have the effect of completely randomizing the order of the data, as well as what data is sampled every epoch.")
parser.add_argument("--use_random_sampler_test_dataset",
                    action="store_true",
                    help="Whether or not to use a RandomSampler for the test dataset. This will have the effect of completely randomizing the order of the data, as well as what data is sampled every epoch.")
parser.add_argument("--train_dataset_num_workers",
                   type=int,
                   default=0,
                   help="Number of training dataset workers.")
parser.add_argument("--val_dataset_num_workers",
                   type=int,
                   default=0,
                   help="Number of validation dataset workers.")
parser.add_argument("--test_dataset_num_workers",
                   type=int,
                   default=0,
                   help="Number of test dataset workers.")
parser.add_argument("--check_test_set_performance",
                    action="store_true",
                    help="Set to calculate test set performance in addition to validation performance.")

# Training parameters.
parser.add_argument("--train_batch_size",
                    type=int,
                    default=None,
                    help="Training batch size.")
parser.add_argument("--val_batch_size",
                    type=int,
                    default=None,
                    help="Validation batch size.")
parser.add_argument("--test_batch_size",
                    type=int,
                    default=None,
                    help="Test batch size.")
parser.add_argument("--n_epochs",
                    type=float,
                    default=None,
                    help="Total (possibly float) number of training epochs.")
parser.add_argument("--evaluation_frequency",
                    type=float,
                    default=None,
                    help="Frequency at which to evaluate model on VALIDATION dataset, can be fractional. Can evaluate on test dataset at most once at the end of training.")
parser.add_argument("--host_gpu_id",
                    type=int,
                    default=None,
                    help="Host GPU ID.")
parser.add_argument("--clip_grad_norm_value",
                    type=float,
                    default=None,
                    help="L2 norm at which to clip gradients of the DLRM.")
parser.add_argument("--optim_type",
                    type=str,
                    choices=["sgd"],
                    default="sgd",
                    help="Optimizer type.")    # DLRM code uses sparse embedding gradients, requiring either SparseAdam or SGD. SparseAdam does not support the dense gradients of the supernet, so SGD is currently the only feasible option.
parser.add_argument("--lr",
                    type=float,
                    default=None,
                    help="Initial learning rate.")
parser.add_argument("--wd",
                    type=float,
                    default=None,
                    help="Weight decay.")
parser.add_argument("--lr_base",
                    type=float,
                    default=None,
                    help="LR = lr * ((lr_base) ** (num_epochs)). Note that this formula may be applied at every training step or every epoch.")
parser.add_argument("--update_lr_every_step",
                    action="store_true",
                    help="If set, LR will be updated every step instead of every epoch.")

# Seed.
parser.add_argument("--seed",
                    type=int,
                    default=1,
                    help="Random seed to ensure results can be replicated. This seed is used for random, numpy, and torch.")

# Needed to interace with tuning script.
parser.add_argument("--save_metrics_param",
                    type=str,
                    default="",
                    help="Path at which to save a file to tell tuning.py that this script is done running.")

# Load from existing checkpoint.
parser.add_argument("--checkpoint",
                    type=str,
                    default="",
                    help="If checkpoint specifies, assumes state dictionary of model can be directly loaded from it and loads.")

# Parse arguments.
args = parser.parse_args()

# Set seed.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Replace argument for emb_card search space. This is probably not necessary.
if args.feature_counts_file:
    args.embeddings_num_vectors = np.load(args.feature_counts_file)["counts"]

# Load the saved architecture information.
saved_arch_info = pickle.load(open(args.saved_arch_filename, "rb"))

# Create the DLRM_Net model.
dlrm = DLRM_Net(**saved_arch_info["arch_config"])

# Load from checkpoint if needed.
if args.checkpoint != "":
    # Load state dictionary.
    saved_state_dictionary = torch.load(args.checkpoint, map_location="cpu")

    # Set model state dictionary.
    loading_result = dlrm.load_state_dict(saved_state_dictionary)

    # Print information.
    print(f"LOADED STATE DICTIONARY FROM CHECKPOINT {args.checkpoint}.")

    # Check if there were any problems.
    if not (len(loading_result.missing_keys) == 0 and len(loading_result.unexpected_keys) == 0):
        print(f"WARNING: MISSING OR UNEXPECTED KEYS IN LOADED STATE DICTIONARY, EXITING. MISSING KEYS = {loading_result.missing_keys}. UNEXPECTED KEYS = {loading_result.unexpected_keys}.")
        exit()

# Get dataloaders.
dataloaders_loaded = False
while not dataloaders_loaded:
    try:
        train_dataloader, val_dataloader, test_dataloader = get_dnas_dataloaders(args, sampled_training=True)
        dataloaders_loaded = True
    except Exception:
        continue

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

    # Exit.
    exit()

# Move DLRM supernet to GPU.
try:
    host_device = torch.device(f"cuda:{args.host_gpu_id}")
    dlrm.to(host_device)
except RuntimeError as oom_error:
    write_oom_exit(oom_error)

# Optimizer class.
optim_class = STR_TO_OPTIM[args.optim_type.lower()]

# Optimizer initialization parameters.
optim_init_params = {"lr" : args.lr, "weight_decay" : args.wd}

# Create optimizer.
optim = optim_class(dlrm.parameters(), **optim_init_params)

# Function which specified how the LR changes during training. Note that
# this function return the RATIO of the current learning rate to the
# initial learning rate, and not the current learning rate itself.
# Code supports multiple optimizer parameter groups hence returning
# an array of LRs.
optim_lr_lambdas = [lambda curr_epoch: (args.lr_base ** curr_epoch)]

# Initial learning rates for the different parameters groups in each
# optimizer. Currently there is only one parameter group used per optimizer,
# however, the code supports multiple parameters groups, each with their own
# initial learning rate and learning rate schedule.
initial_lrs = [args.lr]

# The only loss function currently supported is binary cross-entropy (BCE).
loss_function = nn.BCELoss()

# Store loss over time.
loss_over_time = []

# Number of inner training loops to run.
last_epoch_fractional = (int(args.n_epochs) != args.n_epochs)
n_inner_loops = (int(args.n_epochs) + 1) if last_epoch_fractional else int(args.n_epochs)

# Number of iterations in one epoch.
num_iters_per_epoch = len(train_dataloader)

# Store validation results over time.
val_results = {}

# Generate points at which to run validation
# (assumes we do not evaluate before doing any training.)
eval_epoch_points = []
start_epochs = 0.00
while start_epochs < args.n_epochs:
    start_epochs += args.evaluation_frequency
    eval_epoch_points.append(start_epochs)

# Store the current evaluation point.
curr_eval_ix = 0

# Main DLRM training loop, based heavily on code in dlrm_s_pytorch.py.
for epoch in range(n_inner_loops):
    # If this is the last epoch, then we may need to complete only part of it
    # if args.n_epochs is fractional.
    last_epoch = (epoch == n_inner_loops - 1)

    # Store the losses for this epoch.
    curr_epoch_loss = []

    # Update LR.
    step_lambda_lr(optim, optim_lr_lambdas, epoch, initial_lrs)

    # Training loop for one epoch.
    for iter, (X, lS_o, lS_i, T) in enumerate(train_dataloader):
        # If this is the last batch with incorrect batch size, skip.
        if list(T.size())[0] != train_dataloader.batch_size:
            continue

        # Epochs elapsed.
        epochs_elapsed = epoch + (iter / num_iters_per_epoch)

        # Check if we need to update the LRs every step.
        if args.update_lr_every_step:
            step_lambda_lr(optim, optim_lr_lambdas, epochs_elapsed, initial_lrs)

        # If this is the last epoch and we are done with the
        # fractional part of this epoch that we need to complete,
        # then exit the loop.
        if last_epoch and last_epoch_fractional:
            if epochs_elapsed + (1 / num_iters_per_epoch) > args.n_epochs:
                break

        try:
            # Zero grads.
            optim.zero_grad()

            # Move data to GPU.
            dense_features, sparse_offsets, sparse_indices, labels = \
                    move_data_to_gpu(X, lS_o, lS_i, T, host_device)

            # Forward pass.
            click_probabilities = dlrm(dense_features, sparse_offsets, sparse_indices)
            print(f"Iteration {iter}, click probabilities mean = {torch.mean(click_probabilities)}, stdev = {torch.std(click_probabilities)}")

            # Calculate loss.
            loss = loss_function(click_probabilities, labels)
            print(f"Iteration {iter}, loss = {loss.item()}")

            # Record the current loss.
            curr_epoch_loss.append(loss.item())

            # Backward.
            loss.backward()

            # Clip gradients.
            clip_grad(dlrm.parameters(), args.clip_grad_norm_value)

            # Optimizer step.
            optim.step()

        except RuntimeError as oom_error:
            write_oom_exit(oom_error)

        # Check if we need to get the validation results.
        if epochs_elapsed + (1 / num_iters_per_epoch) >= eval_epoch_points[curr_eval_ix]:
            # Perform evaluation.
            try:
                val_results[eval_epoch_points[curr_eval_ix]] = evaluate(dlrm, val_dataloader)
            except RuntimeError as oom_error:
                write_oom_exit(oom_error)

            # Save result.
            curr_model_state_dict = dlrm.state_dict()
            sampled_arch_results = {"saved_arch_filename" : args.saved_arch_filename,
                                    "saved_arch_info" : saved_arch_info,
                                    "training_loss" : loss_over_time + curr_epoch_loss,
                                    "validation_results" : val_results,
                                    "testing_results" : None,
                                    "saved_arch_state_dict" : curr_model_state_dict}
            torch.save(sampled_arch_results, f"intermediate_{eval_epoch_points[curr_eval_ix]}_{args.save_metrics_param}", _use_new_zipfile_serialization=False)

            # Now that we have performed validation at this epoch point,
            # increment the current evaluation index.
            curr_eval_ix += 1

    # Add curr_epoch_loss to overall loss list.
    loss_over_time.append(curr_epoch_loss)

if args.n_epochs != 0.00:
    # Get validation performance of model now that training is complete.
    # Perform evaluation.
    try:
        val_results[eval_epoch_points[-1]] = evaluate(dlrm, val_dataloader)
    except RuntimeError as oom_error:
        write_oom_exit(oom_error)

    # Save result.
    curr_model_state_dict = dlrm.state_dict()
    sampled_arch_results = {"saved_arch_filename" : args.saved_arch_filename,
                            "saved_arch_info" : saved_arch_info,
                            "training_loss" : loss_over_time,
                            "validation_results" : val_results,
                            "testing_results" : None,
                            "saved_arch_state_dict" : curr_model_state_dict}
    torch.save(sampled_arch_results, f"intermediate_{eval_epoch_points[-1]}_{args.save_metrics_param}", _use_new_zipfile_serialization=False)

# Optionally test model.
if args.check_test_set_performance:
    # Get the test set performance.
    try:
        test_set_performance = evaluate(dlrm, test_dataloader)
    except RuntimeError as oom_error:
        write_oom_exit(oom_error)

    # Save results.
    curr_model_state_dict = dlrm.state_dict()
    sampled_arch_results = {"saved_arch_filename" : args.saved_arch_filename,
                            "saved_arch_info" : saved_arch_info,
                            "training_loss" : loss_over_time,
                            "validation_results" : val_results,
                            "testing_results" : test_set_performance,
                            "saved_arch_state_dict" : curr_model_state_dict}

    # Remove any folders from the saved arch filename.
    processed_saved_arch_filename = args.saved_arch_filename.split("/")[-1]

    torch.save(sampled_arch_results, f"test_set_results_{processed_saved_arch_filename}_{args.save_metrics_param}", _use_new_zipfile_serialization=False)

else:
    sampled_arch_results = {"saved_arch_filename" : args.saved_arch_filename,
                        "saved_arch_info" : saved_arch_info,
                        "training_loss" : loss_over_time,
                        "validation_results" : val_results,
                        "testing_results" : None,
                        "saved_arch_state_dict" : curr_model_state_dict}

# Once the DNAS process is done, in order for tuning.py to know
# that the script is done running, save a file at the save_metrics_param
# location.
torch.save(sampled_arch_results, args.save_metrics_param, _use_new_zipfile_serialization=False)
