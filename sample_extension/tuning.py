#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/24/21

import os
import sys
import numpy as np
import time
import pickle
from queue import deque
import torch
import copy

"""
Hyperparameter tuning script. Runs jobs specified by a hyperparameter search space and launches them on GPUs.
The script always checks if any jobs have been completed, and, if they have, immediately launches another job on the free GPU. This allows it to use the GPUs as efficiently as possible.

Input is a tuning config file which is sys.argv[1], and a global experiment name which is sys.argv[2].

This is a text file with five lines.
The first line is the base command to run the fine tuning.
The second line is the list of arguments with predefined values for non-tunable parameters, and {} notation for tunable parameters with all options (e.g. --lr={0.1,0.2,0.3,0.4,0.5}). The GPU ID hyperparam should be specified as --param_name=GPU_ID_PARAM..
The third line is in the format GPU_IDs:x,y,z where x, y, and z are integer GPU IDs which tuning.py assumes it owns.
The fourth line is in the format NUM_JOBS_PER_GPU:x where x is the number of jobs per GPU to launch.
The fifth line consists of information regarding evaluation of the model: the maximum number of training epochs and the frequency at which to evaluate (e.g. \"5,1\" means tuning over training for 1, 2, 3, 4, and 5 epochs, in addition to other hyperparameters).
"""

def unique_experiment_identifier(chosen_hyperparam_indices):
	"""
	Returns a unique identifier to save the eval metrics.

	chosen_hyperparam_indices is a list of the indices of the hyperparameters in their respective lists that we are using in this configuration.
	"""

	# Store output ID.
	output_id = ""

	# Output is just the indices of the chosen hyperparameters.
	for i, hyperparam_ix in enumerate(chosen_hyperparam_indices):
		# Do not add underscore if we are at the end of the string.
		output_id += str(hyperparam_ix) + ("_" if i != (len(chosen_hyperparam_indices) - 1) else "")

	# Return output ID.
	return output_id

def get_launch_string(chosen_hyperparam_indices, hyperparam_options, gpu_id, save_file_name, base_command, args_string):
	"""
	Returns the string to launch the job.

	chosen_hyperparam_indices is the list of indies of the hyperparameters in this configuration.
	hyperparam_options is a list of lists of the hyperparameters that can be chosen.
	gpu_id is the GPU ID on which to launch the job.
	save_file_name is the name of the file in which to save the evaluation (and other) information when the job is done.
	base_command is the base command to launch the job.
	args_string is the second line of the tuning config file.

	Assumes that there will never be two identical hyperparameter lists for different hyperparameters.
	Assumes that options are read in order.
	"""

	# Basic structure: construct the whole string, construct {} lists, replace {} lists with exact hyperparams from hyperparam_options[chosen_hyperparam_index]. Replace GPU ID param and save file hyperparam with gpu id and config ID with global experiment name. Then add nohup at the beginning and & at the end.
	entire_string = base_command + " " + args_string

	# Construct lists.
	dict_strings = []
	for options in hyperparam_options:
		curr_string = "{"
		for i, option in enumerate(options):
			curr_string += str(option) + (", " if i != (len(options) - 1) else "}")
		dict_strings.append(curr_string)

	# Construct chosen hyperparams.
	chosen_hyperparams = [str(options[index]) for options, index in zip(hyperparam_options, chosen_hyperparam_indices)]

	# Replace lists with hyparparameters.
	for chosen_hyperparam, dict_string in zip(chosen_hyperparams, dict_strings):
		entire_string = entire_string.replace(dict_string, chosen_hyperparam)

	# Replace GPU ID with GPU ID.
	entire_string = entire_string.replace("GPU_ID_PARAM", str(gpu_id))
	entire_string = entire_string.replace("SAVE_METRICS_PARAM", save_file_name)

	# Add nohup and &.
	command_with_nohup = "nohup " + entire_string
	launch_command = command_with_nohup + " &"

	# Return the launch command.
	return launch_command

def jobs_done(save_files):
	"""
	Checks if the save_files exist.
	"""

	# Try opening each file.
	for save_file in save_files:
		# If it exists the jobs may be done running.
		try:
			_ = open(save_file, "rb")

		# If it does not exist the jobs are still running.
		except FileNotFoundError:
			return False

	return True

def get_all_options(args_string):
	"""
	Read all of the hyperparameter options from the arguments string.
	"""

	# Get all the list strings first.
	char_ix = 0
	list_start = -1
	list_strings = []

	# Read the dictionaries which contain the options.
	for char_ix, curr_char in enumerate(args_string):
		if curr_char == "{":
			list_start = char_ix

		if curr_char == "}":
			list_strings.append(args_string[list_start : char_ix + 1])
			list_start = -1

	# Now convert the list strings into options.
	all_options = []
	for list_ix, list_string in enumerate(list_strings):
		# Remove brackets.
		list_str_no_brackets = list_string.split("{")[1].split("}")[0]

		# Get options.
		list_options = list_str_no_brackets.split(", ")

		# Add all all_options.
		all_options.append(list_options)

	# Return all options.
	return all_options

if __name__ == "__main__":
	# Convert strings to different types using these functions.
	str_to_func = {"int" : int, "str" : str, "float" : float}

	# Read the arguments.
	tuning_config_file = sys.argv[1]
	global_experiment_name = sys.argv[2]
	load_intermediate_files = (True if int(sys.argv[3]) == 1 else False)

	# Read everything from tuning_config_file.
	tuning_config_lines = open(tuning_config_file, "r").readlines()
	tuning_config_lines = [tuning_config_line.split("\n")[0] for tuning_config_line in tuning_config_lines]

	# Extract necessary information.
	base_command = tuning_config_lines[0]
	args_string = tuning_config_lines[1]
	all_gpu_ids = [int(possible_id) for possible_id in tuning_config_lines[2].split(":")[1].split(",")]
	num_jobs_per_gpu = int(tuning_config_lines[3].split(":")[1])
	[max_num_epochs, epoch_eval_freq] = [int(value) for value in tuning_config_lines[4].split(":")[1].split(",")]

	# Now figure out after which epochs evaluations will occur.
	evaluation_epochs = []
	start_epochs = 0.00
	while start_epochs < max_num_epochs:
		start_epochs += epoch_eval_freq
		evaluation_epochs.append(start_epochs)

	# Get the lists of potential hyperparameters.
	all_hyperparam_options = get_all_options(args_string)

	# The total number of jobs as well as the number of options for each hyperparameter.
	num_options = [len(options) for options in all_hyperparam_options]
	total_num_jobs = int(np.cumprod(num_options)[-1])

	# Create list of all jobs and GPUs on which to launch them.
	# all_jobs is a list of {"hyperparam_indices" : ..., "gpu_id" : ..., "save_file_name" : ...} dictionaries.

	# Division factors to get hyperparameter option indices from job index.
	open("tuning_logfile", "a").write(f"{all_hyperparam_options}\n")
	open("tuning_logfile", "a").flush()
	division_factors = [1] + list(np.cumprod(num_options))[:-1]
	open("tuning_logfile", "a").write(f"{division_factors}\n")
	open("tuning_logfile", "a").flush()

	# Get all jobs.
	all_jobs = []
	save_file_names = []
	job_configs = []
	for job_index in range(total_num_jobs):
		# Get the information for this job.
		curr_option_indices = [((job_index // division_factors[hyperparam_ix]) % num_options[hyperparam_ix]) for hyperparam_ix in range(len(num_options))]
		save_file_names.append("save_file_" + global_experiment_name + "_" + unique_experiment_identifier(curr_option_indices))

		# Check that the save file name DOES NOT EXIST already.
		if jobs_done([save_file_names[-1]]):
			assert 0 == 1, "ERROR: YOU ARE PROBABLY FORGETTING TO CHANGE THE GLOBAL EXPERIMENT NAME FROM FROM THE LAST EXPERIMENT. SOME OR ALL OF THE SAVE FILES ALREADY EXIST WHICH COULD CAUSE MULTIPLE JOBS TO BE LAUNCHED AT THE SAME TIME WHEN THEY SHOULD NOT BE."

		# Create the job config and add it to the list. Leave the GPU ID undefined for now because we start jobs on GPUs whenever they become available.
		curr_job_config = {"hyperparam_indices" : curr_option_indices, "gpu_id" : None, "save_file_name" : save_file_names[-1]}
		job_configs.append(curr_job_config)

	# Expand the GPU IDs list with the number of jobs per GPU.
	# This expanded_gpu_ids list is useful becuase it allows us to determine which GPU to use for each job when starting the initial set of jobs.
	expanded_gpu_ids = []
	for curr_gpu_id in all_gpu_ids:
      		for job_num in range(num_jobs_per_gpu):
                	expanded_gpu_ids.append(curr_gpu_id)

	# Number of jobs in one batch.
	num_jobs_per_batch = len(expanded_gpu_ids)
	assert num_jobs_per_batch == num_jobs_per_gpu * len(all_gpu_ids), "ERROR: SOMETHING IS WRONG WITH THE CODE."

	# Put all the jobs in a FIFO queue.
	job_q = deque()
	for curr_job in job_configs:
		job_q.append(curr_job)

	# Dictionary of all save files and their corresponding jobs.
	save_files_to_job_configs = {job_config["save_file_name"] : job_config for job_config in job_configs}

	# All save files - for later when we need to wait for all jobs to be done.
	all_save_files_list = [job_config["save_file_name"] for job_config in job_configs]

	# Whether or not the tuning just started.
	tuning_just_started = True

	# Queue for the save file names of jobs we have started.
	running_q = deque()

	# Num launched jobs.
	launched_num = 0

	# When there are no jobs left to launch, we are done.
	while len(list(job_q)) > 0:
		# If the training just started, then don't bother looking for logfiles; instead, start as many jobs as possible.
		if tuning_just_started:
			for curr_gpu_id in expanded_gpu_ids:
				# If the job queue is empty we are done.
				if len(list(job_q)) == 0: break

				# Start a job on this GPU. Use popleft to do a FIFO queue.
				job_config_to_launch = job_q.popleft()

				# Set the GPU ID.
				job_config_to_launch["gpu_id"] = curr_gpu_id

				# Add to running queue.
				curr_job_save_file = job_config_to_launch["save_file_name"]
				running_q.append(curr_job_save_file)

				# Get launch command.
				curr_launch_command = get_launch_string(job_config_to_launch["hyperparam_indices"], all_hyperparam_options, job_config_to_launch["gpu_id"], curr_job_save_file, base_command, args_string)

				# Launch job.
				open("tuning_logfile", "a").write(f"INITIALLY LAUNCHING: {launched_num} {curr_launch_command}.\n")
				open("tuning_logfile", "a").flush()
				os.system(curr_launch_command)
				launched_num += 1

			# Set that tuning did not just start.
			tuning_just_started = False

		# If not search for jobs to start.
		else:
			open("tuning_logfile", "a").write(f"CHECKING FOR JOBS TO START.\n")
			open("tuning_logfile", "a").flush()
			# Go through each save file in the running jobs queue, and if it is there, start another job.
			for check_save_file in list(running_q):
				# Check for two conditions.
				job_completed = jobs_done([check_save_file])
				job_oom = jobs_done([f"oom_error_{check_save_file}"])

				if job_completed or job_oom:
					# If OOM, relaunch the job later.
					if job_oom:
						# Get the original job config.
						original_job_config = save_files_to_job_configs[check_save_file]

						# Add back the original config to the job queue. It will be
						# re-started after all the other jobs have been launched.
						job_q.append(original_job_config)

						# Remove the OOM error file; otherwise, when we relaunch the job
						# later, we will think it has crashed again.
						os.system(f"rm oom_error_{check_save_file}")

						# We do want to execute the rest of the code to launch a new job.

					# There if a job done.
					done_or_crashed_string = "DONE" if job_completed else "CRASHED"
					open("tuning_logfile", "a").write(f"JOB {done_or_crashed_string}: {check_save_file}.\n")
					open("tuning_logfile", "a").flush()
					# Remove the jobs from the running_q. Can use remove since each save file name is unique.
					running_q.remove(check_save_file)

					# Get the job spec that finished.
					just_finished_job = save_files_to_job_configs[check_save_file]

					# Get the GPU on which it was running.
					now_vacant_gpu_id = just_finished_job["gpu_id"]

					# If there are no jobs left to run:
					if len(list(job_q)) == 0:
						open("tuning_logfile", "a").write(f"JOB QUEUE EMPTY WITH JOB {check_save_file} FINISHED.\n")
						open("tuning_logfile", "a").flush()
						# Continue so that we don't run popleft() on an empty queue.
						continue

					# Get the next job to run.
					job_to_start = job_q.popleft()

					# Set the GPU ID.
					job_to_start["gpu_id"] = now_vacant_gpu_id

					# Add to running queue.
					job_to_start_save_file = job_to_start["save_file_name"]
					running_q.append(job_to_start_save_file)

					# Get launch command.
					curr_launch_command = get_launch_string(job_to_start["hyperparam_indices"], all_hyperparam_options, job_to_start["gpu_id"], job_to_start_save_file, base_command, args_string)

					# Launch job.
					open("tuning_logfile", "a").write(f"LAUNCHING: {launched_num} {curr_launch_command}.\n")
					open("tuning_logfile", "a").flush()
					os.system(curr_launch_command)
					launched_num += 1

		# Wait 1 second before checking again.
		time.sleep(1.0)

	# Wait for all of the save files to be written i.e. for all jobs to be done.
	# When this is the case the tuning.py job will finish, but the GPUs should
	# all still be occupied by the jobs launched when the queue was empty.
	while not jobs_done(all_save_files_list):
		# Wait another 5 seconds.
		time.sleep(5.0)

	if load_intermediate_files:
		# Read all of the save files.
		all_metrics_all_jobs_str = ""
		all_metrics_all_jobs = []
		all_metrics_all_jobs_no_checkpoints = []
		for job_config in job_configs:
			# Get all of the save files for this job.
			all_curr_job_save_files = ["intermediate_" + str(potential_eval_epoch) + "_" + job_config["save_file_name"] for potential_eval_epoch in evaluation_epochs]

			# Get information for all intermediate save files as well as the last one.
			for curr_job_save_file in all_curr_job_save_files + [job_config["save_file_name"]]:
				# All job information. Loading evaluation may be from files saved with _use_new_zipfile_serialization=False and torch.save, or pickle.dump.
				all_job_information = job_config
				try:
					all_job_information["evaluation_information"] = torch.load(curr_job_save_file, map_location="cpu")
				except RuntimeError:
					all_job_information["evaluation_information"] = pickle.load(open(curr_job_save_file, "rb"))

				# Remove checkpoints.
				all_job_information_no_checkpoint = copy.deepcopy(all_job_information)
				all_job_information_no_checkpoint["evaluation_information"]["saved_arch_state_dict"] = None

				# Add all job information to all_metrics_all_jobs.
				all_metrics_all_jobs.append(all_job_information)
				all_metrics_all_jobs_no_checkpoints.append(all_job_information_no_checkpoint)

				# Add all job information as a string ao all_metrics_all_jobs_str.
				all_metrics_all_jobs_str += str(all_job_information) + "\n"

	else:
		all_metrics_all_jobs = []
		all_metrics_all_jobs_str = "[]"
		all_metrics_all_jobs_no_checkpoints = []

	# Dump the information.
	with open("all_configs_save_file_" + global_experiment_name, "wb") as jobs_writefile: pickle.dump(all_metrics_all_jobs, jobs_writefile)
	with open("all_configs_no_checkpoints_save_file_" + global_experiment_name, "wb") as no_checkpoints_jobs_writefile: pickle.dump(all_metrics_all_jobs_no_checkpoints, no_checkpoints_jobs_writefile)
	open("all_configs_str_save_file_" + global_experiment_name, "w").write(all_metrics_all_jobs_str)
