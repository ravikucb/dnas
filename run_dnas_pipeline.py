#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/25/21

import os
import argparse
from tuning import jobs_done
import time
from dnas_data_utils import get_dnas_dataloaders
import pickle

# Arguments.
parser = argparse.ArgumentParser("Run DNAS pipeline end-to-end i.e. first supernet training, and then sampled architecture training.")

# Python command argument.
parser.add_argument("--python_cmd",
                    type=str,
                    default="python",
                    help="Command to run python.")

# Dataset arguments needed so that all the jobs can be launched in parallel
# by generating the dataset beforehand. Otherwise, the dataset generation
# fails because each job tries to generate the dataset from the raw data file
# at the same time.
parser.add_argument("--search_space",
                    type=str,
                    default=None,
                    help="Search space.")
parser.add_argument("--embeddings_num_vectors",
                    type=int,
                    nargs="+",
                    help="Number of vectors in each embedding table e.g. \"10000 1000 100\" for a dataset with 3 categorical features. Note that this should include a category for missing categorical feature data.")
parser.add_argument("--dataset",
                    type=str,
                    choices=["kaggle", "terabyte"],
                    default="kaggle",
                    help="Dataset to use; either kaggle or terabyte.")
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
parser.add_argument("--weights_training_proportion",
                    type=float,
                    default=0.8,
                    help="Proportion of dataset after no-click and overall subsampling assigned to train weights. Remainder used to train architecture parameters.")
parser.add_argument("--raw_data_file",
                    type=str,
                    default="",
                    help="Raw dataset file.")
parser.add_argument("--mem_map_weights_data_file",
                    type=str,
                    default="",
                    help="Weights training portion of memory map dataset file.")
parser.add_argument("--mem_map_arch_params_data_file",
                    type=str,
                    default="",
                    help="Architecture parameters training portion of memory map dataset file.")
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
                    help="Data randomization method. Note that data randomization is performed BEFORE splitting into weights/architecture parameters datasets, and that the splitting operation preserves the order of the dataset and thus the randomization method.")
parser.add_argument("--category_dropping",
                    type=str,
                    choices=["none", "modulo", "frequency"],
                    default="frequency",
                    help="Method by which categories in categorical features are dropped. If \"none\" is selected, all categorical indices are kept the same. \"modulo\" will hash the indices of each feature to the number of vectors specified in --embeddings_num_vectors, and \"frequency\" will drop the rarest categories for each features, mapping them to the missing category, such that the number of remaining categories matches the number specified in --embeddings_num_vectors. Note that if \"module\" is specified, the same number of vectors should be allowed for each feature so that the max_ind_range argument to getCriteoAdData can be used.")
parser.add_argument("--use_random_sampler_weights_dataset",
                    action="store_true",
                    help="Whether or not to use a RandomSampler for the weights training dataset. This will have the effect of completely randomizing the ordering of the data, as well as what data is sampled every epoch.")
parser.add_argument("--use_random_sampler_arch_params_dataset",
                    action="store_true",
                    help="Whether or not to use a RandomSampler for the architecture parameters training dataset. This will have the effect of completely randomizing the order of the data, as well as what data is sampled every epoch.")
parser.add_argument("--weights_dataset_num_workers",
                   type=int,
                   default=0,
                   help="Number of weights dataset workers.")
parser.add_argument("--arch_params_dataset_num_workers",
                   type=int,
                   default=0,
                   help="Number of arch params dataset workers.")
parser.add_argument("--weights_batch_size",
                    type=int,
                    default=256,
                    help="Weights training batch size.")
parser.add_argument("--arch_params_batch_size",
                    type=int,
                    default=256,
                    help="Arch params training batch size.")

# Tuning configuration paths and experiment ID.
parser.add_argument("--train_dnas_tuning_config",
                    type=str,
                    default="",
                    help="Path to tuning configuration file for train_dnas.py.")
parser.add_argument("--train_sampled_tuning_config",
                    type=str,
                    default="",
                    help="Path to tuning configuration file for train_sampled.py.")
parser.add_argument("--global_experiment_id",
                    type=str,
                    default="",
                    help="Global experiment ID used for tuning.")

# Parse arguments.
args = parser.parse_args()

# Automatically adjust args.category_dropping by search space.
if args.search_space == "emb_card":
    args.category_dropping = "none"
    print(f"ADJUSTED CATEGORY DROPPING {args.category_dropping}.")

# Before doing anything else, generate the processed data files.
# Note that the only arguments that actually need to be set
# (to anything other than default) for data processing are
# --dataset, --raw_data_file_in, --embeddings_num_vectors,
# and --category_dropping.
_, _ = get_dnas_dataloaders(args, sampled_training=False)

# Add python command to tuning configuration files.
training_dnas_config = open(args.train_dnas_tuning_config, "r").read()
training_dnas_config = training_dnas_config.replace("REPLACE_WITH_PYTHON_CMD", args.python_cmd)
training_sampled_config = open(args.train_sampled_tuning_config, "r").read()
training_sampled_config = training_sampled_config.replace("REPLACE_WITH_PYTHON_CMD", args.python_cmd)

# Put tuning experiment ID in tuning config for train sampled.
training_sampled_config = training_sampled_config.replace("REPLACE_WITH_TUNING_EXPERIMENT_ID", f"{args.global_experiment_id}_sampled")
training_sampled_config = training_sampled_config.replace("REPLACE_WITH_GLOBAL_EXPERIMENT_ID", args.global_experiment_id)

# Write updated tuning configurations.
open(args.train_dnas_tuning_config, "w").write(training_dnas_config)
open(args.train_sampled_tuning_config, "w").write(training_sampled_config)

# Create logfile.
logfile_open = open(f"run_dnas_pipeline_logfile_{args.global_experiment_id}", "w")

# Write info to the logfile.
logfile_open.write(f"STARTED SUPERNET TRAINING TUNING AT {time.time()}\n")
logfile_open.flush()

# Start supernet training tuning.
os.system(f"nohup {args.python_cmd} tuning.py {args.train_dnas_tuning_config} {args.global_experiment_id}_supernet 0")

# Wait until supernet training tuning is complete.
while not jobs_done([f"all_configs_save_file_{args.global_experiment_id}_supernet"]):
    # This means the supernet training tuning has not yet completed.
    time.sleep(10.0)

# Write info the logfile.
logfile_open.write(f"STARTED SAMPLED ARCHITECTURE TRAINING TUNING AT {time.time()}\n")
logfile_open.flush()

# Start sampled architeture training tuning.
os.system(f"nohup {args.python_cmd} tuning.py {args.train_sampled_tuning_config} {args.global_experiment_id}_sampled 1")

# Wait until the sampled architecture training tuning is complete.
while not jobs_done([f"all_configs_save_file_{args.global_experiment_id}_sampled"]):
    # This means the sampled architecture training tuning has not yet completed.
    time.sleep(10.0)

# Write info to logfile.
logfile_open.write(f"FINISHED SAMPLED ARCHITECTURE TRAINING TUNING AT {time.time()}\n")
logfile_open.flush()
