#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/24/21

# Various import statements.
import torch
import torch.nn as nn
from cnn_sampled import ConvNet
from dnas_cnn_data_utils import CNNDataset
import argparse
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
def move_data_to_gpu(images, labels, device):
    return images.to(device), labels.to(device)

# Evaluation function.
def evaluate(cnn, evaluation_dataloader):
    """
    Evaluates CNN on evaluation_dataloader dataset. This can
    be either the validation of test dataset.

    Returns CrossEntropy loss and accuracy on the dataset.
    """

    # In order for model training results to be the same
    # regardless of whether we evaluate, we need to save the
    # CPU and CUDA RNG states before evaluation and load them
    # back after evaluation.
    original_cpu_rng_state = torch.get_rng_state()
    original_gpu_rng_state = torch.cuda.get_rng_state()

    # Create loss function.
    loss_function = nn.CrossEntropyLoss()

    # Store the total loss and number of data points.
    total_loss = 0.0
    num_data_points = 0
    correct_classifications = 0

    for iter, (images, labels) in enumerate(evaluation_dataloader):
        # Move data to GPU.
        images, labels = move_data_to_gpu(images, labels, host_device)

        # Forward pass.
        logits = cnn(images)

        # Calcuate loss.
        loss = loss_function(logits, labels)

        # Add to total loss, update total data points.
        curr_batch_size = list(labels.size())[0]
        total_loss += (loss.item()) * curr_batch_size
        num_data_points += curr_batch_size

        # Update correct classifications.
        correct_classifications += float(torch.sum(torch.argmax(logits, axis=1) == labels))

    # Calculate overall loss and calibration.
    dataset_loss = total_loss / num_data_points

    # Re-load original RNG states.
    torch.set_rng_state(original_cpu_rng_state)
    torch.cuda.set_rng_state(original_gpu_rng_state)

    # Return results.
    return {"loss" : dataset_loss, "accuracy" : correct_classifications / num_data_points}

# Create argument parser.
parser = argparse.ArgumentParser(description="Run sampled CNN training.")

parser.add_argument("--saved_arch_filename",
                    type=str,
                    default="",
                    help="Saved architecture path.")
parser.add_argument("--tuning_experiment_id",
                    type=str,
                    default="",
                    help="Tuning experiment ID used for saving results files.")

# Test set flag.
parser.add_argument("--check_test_set_performance",
                    action="store_true",
                    help="Set to calculate test set performance in addition to validation performance.")

# Training parameters.
parser.add_argument("--train_batch_size",
                    type=int,
                    default=256,
                    help="Training batch size.")
parser.add_argument("--val_batch_size",
                    type=int,
                    default=256,
                    help="Validation batch size.")
parser.add_argument("--test_batch_size",
                    type=int,
                    default=256,
                    help="Test batch size.")
parser.add_argument("--n_epochs",
                    type=float,
                    default=4.0,
                    help="Total (possibly float) number of training epochs.")
parser.add_argument("--evaluation_frequency",
                    type=float,
                    default=1.0,
                    help="Frequency at which to evaluate model on VALIDATION dataset, can be fractional. Can evaluate on test dataset at most once at the end of training.")
parser.add_argument("--host_gpu_id",
                    type=int,
                    default=None,
                    help="Host GPU ID.")
parser.add_argument("--clip_grad_norm_value",
                    type=float,
                    default=100.0,
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
                    default=0.00,
                    help="Weight decay.")
parser.add_argument("--lr_base",
                    type=float,
                    default=0.9,
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

# Load the saved architecture information.
saved_arch_info = pickle.load(open(args.saved_arch_filename, "rb"))

# Create the ConvNet model.
cnn = ConvNet(**saved_arch_info["arch_config"])

# Load from checkpoint if needed.
if args.checkpoint != "":
    # Load state dictionary.
    saved_state_dictionary = torch.load(args.checkpoint, map_location="cpu")

    # Set model state dictionary.
    loading_result = cnn.load_state_dict(saved_state_dictionary)

    # Print information.
    print(f"LOADED STATE DICTIONARY FROM CHECKPOINT {args.checkpoint}.")

    # Check if there were any problems.
    if not (len(loading_result.missing_keys) == 0 and len(loading_result.unexpected_keys) == 0):
        print(f"WARNING: MISSING OR UNEXPECTED KEYS IN LOADED STATE DICTIONARY, EXITING. MISSING KEYS = {loading_result.missing_keys}. UNEXPECTED KEYS = {loading_result.unexpected_keys}.")
        exit()

# Get dataloaders.
train_dataloader, val_dataloader, test_dataloader = torch.utils.data.DataLoader(CNNDataset("train"), batch_size=args.train_batch_size), torch.utils.data.DataLoader(CNNDataset("val"), batch_size=args.val_batch_size), torch.utils.data.DataLoader(CNNDataset("test"), batch_size=args.test_batch_size)

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
try:
    host_device = torch.device(f"cuda:{args.host_gpu_id}")
    cnn.to(host_device)
except RuntimeError as oom_error:
    write_oom_exit(oom_error)

# Optimizer class.
optim_class = STR_TO_OPTIM[args.optim_type.lower()]

# Optimizer initialization parameters.
optim_init_params = {"lr" : args.lr, "weight_decay" : args.wd}

# Create optimizer.
optim = optim_class(cnn.parameters(), **optim_init_params)

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
loss_function = nn.CrossEntropyLoss()

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
    for iter, (images, labels) in enumerate(train_dataloader):
        # If the batch size is incorrec (e.g. last batch), skip the iteration.
        if list(images.size())[0] != args.train_batch_size:
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
            images, labels = move_data_to_gpu(images, labels, host_device)

            # Forward pass.
            logits = cnn(images)

            # Calculate loss.
            loss = loss_function(logits, labels)

            # Record the current loss.
            curr_epoch_loss.append(loss.item())

            # Backward.
            loss.backward()

            # Clip gradients.
            clip_grad(cnn.parameters(), args.clip_grad_norm_value)

            # Optimizer step.
            optim.step()

        except RuntimeError as oom_error:
            write_oom_exit(oom_error)

        # Check if we need to get the validation results.
        if epochs_elapsed + (1 / num_iters_per_epoch) >= eval_epoch_points[curr_eval_ix]:
            # Perform evaluation.
            try:
                val_results[eval_epoch_points[curr_eval_ix]] = evaluate(cnn, val_dataloader)
            except RuntimeError as oom_error:
                write_oom_exit(oom_error)

            # Save result.
            curr_model_state_dict = cnn.state_dict()
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
        val_results[eval_epoch_points[-1]] = evaluate(cnn, val_dataloader)
    except RuntimeError as oom_error:
        write_oom_exit(oom_error)

    # Save result.
    curr_model_state_dict = cnn.state_dict()
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
    test_set_performance = evaluate(cnn, test_dataloader)

    # Save results.
    curr_model_state_dict = cnn.state_dict()
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
