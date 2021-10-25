#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/29/21

import pickle
import numpy as np
import argparse

# Only argument is file to compare to expected_output.
parser = argparse.ArgumentParser("Verify output of sample DNAS extension.")
parser.add_argument("--output_to_verify",
                    type=str,
                    default=None,
                    help="Path to all configs no_checkpoints file for sampled architectures.")

args = parser.parse_args()

# Load both files.
expected_output = pickle.load(open("expected_output", "rb"))
to_verify_output = pickle.load(open(args.output_to_verify, "rb"))

# Compare training losses.
expected_output_dictionary = {curr_output["save_file_name"] : curr_output["evaluation_information"]["training_loss"] for curr_output in expected_output}
to_verify_output_dictionary = {curr_output["save_file_name"] : curr_output["evaluation_information"]["training_loss"] for curr_output in to_verify_output}

results_verified = True
for save_file_name, expected_loss in expected_output_dictionary.items():
    # Get the loss to veriy.
    loss_to_verify = to_verify_output_dictionary[save_file_name]

    # Convert both loss arrays to numpy 1D arrays.
    expected_loss_np = np.ndarray.flatten(np.array(expected_loss))
    to_verify_np = np.ndarray.flatten(np.array(loss_to_verify))

    # Compute relative error.
    relative_error = np.abs(to_verify_np - expected_loss_np)/expected_loss_np

    # Make sure all errors are low enough.
    for i, curr_error in enumerate(list(relative_error)):
        if curr_error > 1e-4:
            print(f"INCONSISTENCY FOUND WITH RELATIVE ERROR {curr_error} FOR FILE {save_file_name} WITH EXPECTED LOSS {expected_loss_np[i]} AND LOSS TO VERIFY {to_verify_np[i]}.")
            results_verified = False

if results_verified:
    print(f"RESULTS VERIFIED!")
else:
    print(f"DUE TO THE INCONSISTENCIES LISTED ABOVE, RESULTS NOT VERIFIED.")
