#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/21/24

import os
import argparse
from tuning import jobs_done
import time
import pickle

# Arguments.
parser = argparse.ArgumentParser("Run DNAS pipeline end-to-end i.e. first supernet training, and then sampled architecture training.")

# Python command argument.
parser.add_argument("--python_cmd",
                    type=str,
                    default="python",
                    help="Command to run python.")

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
while not jobs_done([f"all_configs_save_file_{args.global_experiment_id}_supernet"]):
    # This means the sampled architecture training tuning has not yet completed.
    time.sleep(10.0)

# Write info to logfile.
logfile_open.write(f"FINISHED SAMPLED ARCHITECTURE TRAINING TUNING AT {time.time()}\n")
logfile_open.flush()
