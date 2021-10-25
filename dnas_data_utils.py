#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Ravi Krishna 07/25/21

# Import statements.
import torch
from torch.utils.data import DataLoader, RandomSampler
from data_utils import getCriteoAdData
from dlrm_data_pytorch import CriteoDataset, collate_wrapper_criteo
import time
import random
from utils import KAGGLE_DAYS
import numpy as np
import os
from os import path

def wrapper_get_criteo_ad_data(dnas_args):
    """
    Wrapper function which takes train_dnas.py args and reads and randomizes
    dataset with getCriteoAdData. getCriteoAdData needs to be extended to drop
    categories not just via modulo but also by dropping infrequent categories.

    Note that this function is only useful if the raw data file is specified
    If the processed input data is directly specified, then this function
    should not be called.
    """

    # Prepare misc arguments to getCriteoAdData.

    # Intermediate output filename.
    output_filename = dnas_args.processed_data_file

    # Get max_ind_range argument.
    num_vectors = list(map(int, dnas_args.embeddings_num_vectors))
    max_ind_range_arg = num_vectors[0] if dnas_args.category_dropping == "modulo" else -1
    print(f"IN wrapper_get_criteo_ad_data, num_vectors = {num_vectors}, max_ind_range_arg = {max_ind_range_arg}")

    # Get days argument.
    days_arg = KAGGLE_DAYS

    # Call getCriteoAdData as done in dlrm_data_pytorch.CriteoDataset.__init__().
    file = getCriteoAdData(
        dnas_args.raw_data_file,
        output_filename,
        max_ind_range_arg,
        dnas_args.no_click_subsample_rate,
        days_arg,
        "none",
        dnas_args.data_randomization,
        dnas_args.dataset == "kaggle",
        dnas_args.memory_map,
        dnas_args.dataset_multiprocessing
    )

    # Return the processed data file.
    return file

def full_subsampling_and_split_mem_map(dnas_args, weights_arch_params_split=True):
    """
    Function to perform additional subsampling of data (no-click and also click)
    and optionally split pre-processed training dataset into weights and arch params portions.
    """

    # We need to process the individual files in the folders.
    # Much of this code is based on code in data_loader_terabyte.

    # Filename prefix.
    raw_data_dir = dnas_args.raw_data_file.split("/")[0]
    filename_prefix = dnas_args.raw_data_file.split("/")[-1].split(".")[0]
    output_filename_prefix = dnas_args.processed_data_file.split("/")[-1]

    # Number of dataset days.
    num_days = KAGGLE_DAYS

    # Get all of the training files.
    training_filenames = []
    raw_training_filenames = []
    for day in range(num_days):
        training_filenames.append(f"{raw_data_dir}/{filename_prefix}_day_{day}_reordered.npz")
        raw_training_filenames.append(f"{raw_data_dir}/{filename_prefix}_day_{day}")

    # Create output directories if needed.
    if weights_arch_params_split:
        weights_dir = path.dirname(dnas_args.mem_map_weights_data_file)
        arch_params_dir = path.dirname(dnas_args.mem_map_arch_params_data_file)

        os.system(f"mkdir {weights_dir}")
        os.system(f"mkdir {arch_params_dir}")
    else:
        # Create processed directory.
        processed_dir = path.dirname(dnas_args.processed_data_file)

        os.system(f"mkdir {processed_dir}")

    # Load the training data as X_int, X_cat, and y for each file.
        # Run full_subsampling_and_split_direct for each file.
        # Write back to output directory.
    total_per_file_weights = []
    total_per_file_arch_params = []
    total_per_file_subsampled_only = []
    for day_ix, (training_filename, raw_training_filename) in enumerate(zip(training_filenames, raw_training_filenames)):
        # Load the data.
        with np.load(training_filename) as curr_data:
            curr_X_int = curr_data["X_int"]
            curr_X_cat = curr_data["X_cat"]
            curr_y = curr_data["y"]

        # Subsample and optionally split.
        if weights_arch_params_split:
            weights_subsampled_train_X_int, weights_subsampled_train_X_cat, \
                weights_subsampled_train_y, arch_params_subsampled_train_X_int, \
                arch_params_subsampled_train_X_cat, arch_params_subsampled_train_y, \
                weights_string, arch_params_string = \
                full_subsampling_and_split_direct(dnas_args, curr_X_int, curr_X_cat, curr_y,
                        weights_arch_params_split=weights_arch_params_split, raw_lines=open(raw_training_filename).read())

            # Update total_per_file_weights and total_per_file_arch_params.
            total_per_file_weights.append(len(weights_subsampled_train_y))
            total_per_file_arch_params.append(len(arch_params_subsampled_train_y))

            # Save data.
            np.savez_compressed(f"{weights_dir}/{filename_prefix}_day_{day_ix}_reordered.npz",
                    X_int=weights_subsampled_train_X_int,
                    X_cat=weights_subsampled_train_X_cat,
                    y=weights_subsampled_train_y)

            np.savez_compressed(f"{arch_params_dir}/{filename_prefix}_day_{day_ix}_reordered.npz",
                    X_int=arch_params_subsampled_train_X_int,
                    X_cat=arch_params_subsampled_train_X_cat,
                    y=arch_params_subsampled_train_y)

            # Save raw data.
            with open(f"{weights_dir}/{filename_prefix}_day_{day_ix}", "w") as weights_writefile:
                weights_writefile.write(weights_string)

            with open(f"{arch_params_dir}/{filename_prefix}_day_{day_ix}", "w") as arch_params_writefile:
                arch_params_writefile.write(arch_params_string)

        else:
            subsampled_train_X_int, subsampled_train_X_cat, subsampled_train_y, subsampled_lines = \
                    full_subsampling_and_split_direct(dnas_args, curr_X_int, curr_X_cat, curr_y,
                        weights_arch_params_split=weights_arch_params_split, raw_lines=open(raw_training_filename).read())

            # Update total_per_file_subsampled.
            total_per_file_subsampled_only.append(len(subsampled_train_y))

            # Save data.
            np.savez_compressed(f"{processed_dir}/{output_filename_prefix}_day_{day_ix}_reordered.npz",
                    X_int=subsampled_train_X_int,
                    X_cat=subsampled_train_X_cat,
                    y=subsampled_train_y)

            # Save raw data.
            with open(f"{processed_dir}/{output_filename_prefix}_day_{day_ix}", "w") as procesed_writefile:
                processed_writefile.write(subsampled_lines)

    # Add day counts file(s).
    if weights_arch_params_split:
        np.savez_compressed(f"{weights_dir}/{filename_prefix}_day_count.npz",
                        total_per_file=total_per_file_weights)

        np.savez_compressed(f"{arch_params_dir}/{filename_prefix}_day_count.npz",
                        total_per_file=total_per_file_arch_params)

        os.system(f"cp {raw_data_dir}/{filename_prefix}_fea_count.npz {weights_dir}/{filename_prefix}_fea_count.npz")
        os.system(f"cp {raw_data_dir}/{filename_prefix}_fea_count.npz {arch_params_dir}/{filename_prefix}_fea_count.npz")

    else:
        np.savez_compressed(f"{processed_dir}/{output_filename_prefix}_day_count.npz",
                        total_per_file=total_per_file_subsampled_only)

        os.system(f"cp {processed_data_dir}/{filename_prefix}_fea_count.npz {arch_params_dir}/{filename_prefix}_fea_count.npz")

    # Don't really need to return anything since all
    # of the data has been written to the folders.
    return f"{weights_dir}/{filename_prefix}", f"{arch_params_dir}/{filename_prefix}"

def combine_lines(lines_list):
    """
    Combines a list of lines into one string with newlines.
    """

    # Store the result string.
    result_string = ""
    for line in lines_list:
        result_string += (line + "\n")

    # Leaves an additional newline at the end which is the same as in the original files.
    return result_string

def full_subsampling_and_split_direct(dnas_args,
                                    train_X_int,
                                    train_X_cat,
                                    train_y,
                                    raw_lines=None,
                                    weights_arch_params_split=True):
    """
    Function to perform additional subsampling of data (no-click and also click)
    and optinally split pre-processed training dataset into weights and arch params portions.
    """

    # Get raw lines to split and subsample.
    if raw_lines: raw_lines_list = raw_lines.split("\n")[:-1]

    # Subsampling.

    # Decide which points to keep.
    keep_point = lambda point: (False if (random.uniform(0, 1) < dnas_args.overall_subsample_rate) else True)
    subsampling_mask = list(map(keep_point, list(range(len(train_y)))))

    # Remove points as needed.
    subsampled_train_X_int = [train_X_int[i] for i in range(len(train_X_int)) if subsampling_mask[i]]
    subsampled_train_X_cat = [train_X_cat[i] for i in range(len(train_X_cat)) if subsampling_mask[i]]
    subsampled_train_y = [train_y[i] for i in range(len(train_y)) if subsampling_mask[i]]
    if raw_lines: subsampled_raw_lines = [raw_lines_list[i] for i in range(len(raw_lines_list)) if subsampling_mask[i]]

    # Split into weights and architecture parameters points if needed.
    if weights_arch_params_split:
        # Decide which points should go to the weights dataset and which
        # points should go to the architecture parameters dataset.
        point_dataset = lambda point: ("weights" if (random.uniform(0, 1) < dnas_args.weights_training_proportion) else "arch_params")
        dataset_assignments = list(map(point_dataset, list(range(len(train_y)))))

        # Split the datasets.
        weights_subsampled_train_X_int = [subsampled_train_X_int[i] for i in range(len(subsampled_train_X_int)) if dataset_assignments[i] == "weights"]
        arch_params_subsampled_train_X_int = [subsampled_train_X_int[i] for i in range(len(subsampled_train_X_int)) if dataset_assignments[i] == "arch_params"]

        weights_subsampled_train_X_cat = [subsampled_train_X_cat[i] for i in range(len(subsampled_train_X_cat)) if dataset_assignments[i] == "weights"]
        arch_params_subsampled_train_X_cat = [subsampled_train_X_cat[i] for i in range(len(subsampled_train_X_cat)) if dataset_assignments[i] == "arch_params"]

        weights_subsampled_train_y = [subsampled_train_y[i] for i in range(len(subsampled_train_y)) if dataset_assignments[i] == "weights"]
        arch_params_subsampled_train_y = [subsampled_train_y[i] for i in range(len(subsampled_train_y)) if dataset_assignments[i] == "arch_params"]

        if raw_lines:
            weights_subsampled_train_lines = [subsampled_raw_lines[i] for i in range(len(subsampled_raw_lines)) if dataset_assignments[i] == "weights"]
            arch_params_subsampled_train_lines = [subsampled_raw_lines[i] for i in range(len(subsampled_raw_lines)) if dataset_assignments[i] == "arch_params"]

        # Return results.
        if raw_lines:
            return weights_subsampled_train_X_int, weights_subsampled_train_X_cat, \
                    weights_subsampled_train_y, arch_params_subsampled_train_X_int, \
                    arch_params_subsampled_train_X_cat, arch_params_subsampled_train_y, \
                    combine_lines(weights_subsampled_train_lines), combine_lines(arch_params_subsampled_train_lines)
        else:
            return weights_subsampled_train_X_int, weights_subsampled_train_X_cat, \
                    weights_subsampled_train_y, arch_params_subsampled_train_X_int, \
                    arch_params_subsampled_train_X_cat, arch_params_subsampled_train_y

    # Return results.
    if raw_lines:
        return subsampled_train_X_int, subsampled_train_X_cat, subsampled_train_y, combine_lines(subsampled_raw_lines)
    else:
        return subsampled_train_X_int, subsampled_train_X-cat, subsampled_train_y

def full_subsampling_and_split(dnas_args,
                                train_X_int=None,
                                train_X_cat=None,
                                train_y=None,
                                weights_arch_params_split=True):
    """
    Function to perform additional subsampling of data (no-click and also click)
    and optinally split pre-processed training dataset into weights and arch params portions.
    """

    # Call the correct function depending on whether we are processing the Terabyte dataset
    # or not.
    if not dnas_args.memory_map:
        return full_subsampling_and_split_direct(dnas_args,
                                train_X_int=train_X_int,
                                train_X_cat=train_X_cat,
                                train_y=train_y,
                                weights_arch_params_split=weights_arch_params_split)
    else:
        return full_subsampling_and_split_mem_map(dnas_args,
                                weights_arch_params_split=weights_arch_params_split)

def get_dnas_dataloaders(dnas_args, sampled_training=False):
    """
    Overall wrapper function which takes
    train_dnas.py arguments and loads from
    an existing dataset or starts the dataset
    processing pipeline with the above
    functions, returning the dataloaders for
    the weights and architecture parameters
    datasets. Creates a CriteoDataset instance
    with the pre-processed data.

    In the below comment, by \"dataset\",
    we mean both the weights training and
    architecture parameters training dataset.

    If the pre-processed dataset already exists,
    directly loads from that dataset and using
    dlrm_data_pytorch.CriteoDatset returns the
    dataloaders.

    If not, then get_dnas_dataloaders will start
    the pipeline by calling the wrapper function
    of getCriteoAdData, then the additional
    pre-processing function, and then using
    dlrm_data_pytorch.CriteoDataset to return
    the dataloaders.

    If sampled_training is False, then
    get_dnas_dataloaders assumes it is being
    called to generate dataloaders to train a
    sampled architecture and so passes
    weights_arch_params_split=False to
    full_subsampling_and_split.
    """
    print(f"Raw data file = {dnas_args.raw_data_file}, processed data file = {dnas_args.processed_data_file}")

    # Get processed data.
    if not dnas_args.load_processed:
        print(f"NOT LOAD PROCESSED")
        _ = wrapper_get_criteo_ad_data(dnas_args)
    else:
        print(f"LOADING PROCESSED DATA.")

    processed_data_file = dnas_args.processed_data_file
    print(f"Setting processed data file = {processed_data_file}")
    print(f"Set processed data file done at time = {time.time()}, args = {dnas_args.__dict__}")

    # Get max_ind_range argument.
    num_vectors = list(map(int, dnas_args.embeddings_num_vectors))
    max_ind_range = num_vectors[0] if dnas_args.category_dropping == "modulo" else -1
    print(f"FETCHED num_vectors = {num_vectors}, max_ind_range = {max_ind_range}.")

    # If sampled_training, create train, val, and test CriteoDatasets and dataloaders.
    if sampled_training:
        print(f"SAMPLED TRAINING LOADING DATA.")
        # Currently does not support entire-dataset subsampling
        # but could easily in the future by calling full_subsampling_and_split
        # with weights_arch_params_split=False.

        # Create datasets.
        train_dataset = CriteoDataset(
                                   dnas_args.dataset,
                                   max_ind_range,
                                   dnas_args.no_click_subsample_rate,
                                   dnas_args.data_randomization,
                                   "train",
                                   dnas_args.raw_data_file,
                                   processed_data_file,
                                   dnas_args.memory_map,
                                   dnas_args.dataset_multiprocessing)

        val_dataset = CriteoDataset(
                                   dnas_args.dataset,
                                   max_ind_range,
                                   0.00,
                                   dnas_args.data_randomization,
                                   "val",
                                   dnas_args.raw_data_file,
                                   processed_data_file,
                                   dnas_args.memory_map,
                                   dnas_args.dataset_multiprocessing)

        test_dataset = CriteoDataset(
                                   dnas_args.dataset,
                                   max_ind_range,
                                   0.00,
                                   dnas_args.data_randomization,
                                   "test",
                                   dnas_args.raw_data_file,
                                   processed_data_file,
                                   dnas_args.memory_map,
                                   dnas_args.dataset_multiprocessing)

        # Use regular DataLoader.
        train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=dnas_args.train_batch_size,
                            shuffle=False,
                            num_workers=dnas_args.train_dataset_num_workers,
                            collate_fn=collate_wrapper_criteo,
                            pin_memory=False,
                            drop_last=False,
                            sampler=(RandomSampler(train_dataset) if dnas_args.use_random_sampler_train_dataset else None))

        val_loader = torch.utils.data.DataLoader(val_dataset,
                            batch_size=dnas_args.val_batch_size,
                            shuffle=False,
                            num_workers=dnas_args.val_dataset_num_workers,
                            collate_fn=collate_wrapper_criteo,
                            pin_memory=False,
                            drop_last=False,
                            sampler=(RandomSampler(val_dataset) if dnas_args.use_random_sampler_val_dataset else None))

        test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=dnas_args.test_batch_size,
                            shuffle=False,
                            num_workers=dnas_args.test_dataset_num_workers,
                            collate_fn=collate_wrapper_criteo,
                            pin_memory=False,
                            drop_last=False,
                            sampler=(RandomSampler(test_dataset) if dnas_args.use_random_sampler_test_dataset else None))

        # Return the DataLoaders.
        return train_loader, val_loader, test_loader

    # Otherwise, create weights and arch params CriteoDatasets by first,
    # creating two training datasets and then replacing the data used,
    # for these two CriteoDataset objects.
    else:
        # Create dataloaders.
        if not dnas_args.memory_map:
            # Create datasets.
            print(f"Weights dataset = {dnas_args.dataset}, raw data file = {dnas_args.raw_data_file}, processed data file = {processed_data_file}.")
            weights_dataset = CriteoDataset(
                                       dnas_args.dataset,
                                       max_ind_range,
                                       dnas_args.no_click_subsample_rate,
                                       dnas_args.data_randomization,
                                       "train",
                                       dnas_args.raw_data_file,
                                       processed_data_file,
                                       dnas_args.memory_map,
                                       dnas_args.dataset_multiprocessing)

            arch_params_dataset = CriteoDataset(
                                       dnas_args.dataset,
                                       max_ind_range,
                                       dnas_args.no_click_subsample_rate,
                                       dnas_args.data_randomization,
                                       "train",
                                       dnas_args.raw_data_file,
                                       processed_data_file,
                                       dnas_args.memory_map,
                                       dnas_args.dataset_multiprocessing)

            # Full-dataset subsampling and weights/arch params split.
            weights_subsampled_train_X_int, weights_subsampled_train_X_cat, weights_subsampled_train_y, \
                arch_params_subsampled_train_X_int, arch_params_subsampled_train_X_cat, arch_params_subsampled_train_y = \
                full_subsampling_and_split(dnas_args, train_X_int=weights_dataset.X_int, train_X_cat=weights_dataset.X_cat,
                    train_y=weights_dataset.y, weights_arch_params_split=True)

            # Replace data.
            weights_dataset.X_int = weights_subsampled_train_X_int
            weights_dataset.X_cat = weights_subsampled_train_X_cat
            weights_dataset.y = weights_subsampled_train_y

            arch_params_dataset.X_int = arch_params_subsampled_train_X_int
            arch_params_dataset.X_cat = arch_params_subsampled_train_X_cat
            arch_params_dataset.y = arch_params_subsampled_train_y

        else:
            # Split data if needed.
            print(f"Load processed = {dnas_args.load_processed}")
            if not dnas_args.load_processed:
                weights_training_file, arch_params_training_file = full_subsampling_and_split(dnas_args, weights_arch_params_split=True)
            else:
                filename_prefix = dnas_args.raw_data_file.split("/")[-1]
                weights_dir = path.dirname(dnas_args.mem_map_weights_data_file)
                arch_params_dir = path.dirname(dnas_args.mem_map_arch_params_data_file)
                weights_training_file, arch_params_training_file = f"{weights_dir}/{filename_prefix}", f"{arch_params_dir}/{filename_prefix}"

            # Create datasets.
            weights_dataset = CriteoDataset(
                                       dnas_args.dataset,
                                       max_ind_range,
                                       dnas_args.no_click_subsample_rate,
                                       dnas_args.data_randomization,
                                       "train",
                                       weights_training_file,
                                       processed_data_file,
                                       dnas_args.memory_map,
                                       dnas_args.dataset_multiprocessing)

            arch_params_dataset = CriteoDataset(
                                       dnas_args.dataset,
                                       max_ind_range,
                                       dnas_args.no_click_subsample_rate,
                                       dnas_args.data_randomization,
                                       "train",
                                       arch_params_training_file,
                                       processed_data_file,
                                       dnas_args.memory_map,
                                       dnas_args.dataset_multiprocessing)

        # Use regular DataLoader.
        weights_loader = torch.utils.data.DataLoader(weights_dataset,
                            batch_size=dnas_args.weights_batch_size,
                            shuffle=False,
                            num_workers=dnas_args.weights_dataset_num_workers,
                            collate_fn=collate_wrapper_criteo,
                            pin_memory=False,
                            drop_last=False,
                            sampler=(RandomSampler(weights_dataset) if dnas_args.use_random_sampler_weights_dataset else None))

        arch_params_loader = torch.utils.data.DataLoader(arch_params_dataset,
                            batch_size=dnas_args.arch_params_batch_size,
                            shuffle=False,
                            num_workers=dnas_args.arch_params_dataset_num_workers,
                            collate_fn=collate_wrapper_criteo,
                            pin_memory=False,
                            drop_last=False,
                            sampler=(RandomSampler(arch_params_dataset) if dnas_args.use_random_sampler_arch_params_dataset else None))

        # Return the DataLoaders.
        return weights_loader, arch_params_loader
