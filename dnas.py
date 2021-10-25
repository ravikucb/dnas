# Ravi Krishna 09/07/21

import os

"""
Differentiable Neural Architecture Search API

Allows for launching experiments for the 3 search space groups (MLP search, embedding dimension search, and embedding cardinality search) that have already been implemented. Currently,
only runs with the Criteo Kaggle dataset used in the paper.

For a given search space group, uses arguments corresponding to the default tuning configurations for that search space group. Any CHANGES to such arguments should be passed as a
dictionary of arguments {"--arg=123" : "--arg=456"}. String substitutions are performed directly, so any errors in this input dictionary will likely cause errors when the DNAS pipeline is run.

Note that the list of GPU IDs to use for the DNAS training, as well as the list for the sampled architecture training, must always be provided. This is to ensure that the user does
not select GPUs which are not available to be run.

Such lists of GPU IDs are, as with any argument changes, to be directly string-substituted. Hence, they should be formatted as: 0,1,2,3 or 0,1,2,6,7.
"""

# Existing tuning configuration files.
existing_files = {"top_bottom_mlps" : {"dnas" : "mlp_search_train_dnas_tuning_config", "sampled" : "mlp_search_train_sampled_tuning_config"},
                  "emb_dim" : {"dnas" : "embedding_dimension_train_dnas_tuning_config", "sampled" : "embedding_dimension_train_sampled_tuning_config"},
                  "emb_card" : {"dnas" : "embedding_cardinality_train_dnas_tuning_config", "sampled" : "embedding_cardinality_train_sampled_tuning_config"}}

def run_pipeline(search_space_group,
                 config_suffix_experiment_id,
                 train_dnas_gpus,
                 train_sampled_gpus,
                 dnas_num_jobs_per_gpu,
                 sampled_num_jobs_per_gpu,
                 dnas_arg_changes,
                 sampled_arg_changes,
                 python_cmd,
                 memory_map,
                 kaggle_raw_file,
                 mem_map_weights_folder="folder",
                 mem_map_arch_params_folder="folder",
                 load_processed_data=False):

    """
    Runs the DNAS pipeline for the selected search space group, substituting the
    directly specified arguments such as train_dnas_gpus, and substituting the
    dictionary-specified arguments (dnas_arg_changes and sampled_arg_changes).

    DNAS pipeline launched by calling run_kaggle_jobs.sh. Note that the value of memory_map passed to this
    function must match the value of memory_map contained in the tuning config files. The user is responsible
    for making sure this is the case.

    config_suffix_experiment_id is added to the end of the default tuning config filenames to distinguish the new
    experiment configurations from those default configurations and is assumed to be the experiment ID.

    Various other arguments are based directly on arguments to run_kaggle_jobs.sh script.
    """

    # Get default config file names.
    default_dnas_filename = "tuning_configs/" + existing_files[search_space_group]["dnas"]
    default_sampled_filename = "tuning_configs/" + existing_files[search_space_group]["sampled"]

    # Generate new filenames.
    new_dnas_filename = f"{default_dnas_filename}_{config_suffix_experiment_id}"
    new_sampled_filename = f"{default_sampled_filename}_{config_suffix_experiment_id}"

    # Create new config files.
    os.system(f"cp {default_dnas_filename} {new_dnas_filename}")
    os.system(f"cp {default_sampled_filename} {new_sampled_filename}")

    # Get the file text as a string.
    dnas_config = open(new_dnas_filename, "r").read()
    sampled_config = open(new_sampled_filename, "r").read()

    # Replace the GPU ID arguments.
    dnas_config = dnas_config.replace("GPU_IDs:0,1,2,3,4,5,6,7", f"GPU_IDs:{train_dnas_gpus}")
    sampled_config = sampled_config.replace("GPU_IDs:0,1,2,3,4,5,6,7", f"GPU_IDs:{train_sampled_gpus}")

    # Replace the number of jobs arguments.
    dnas_config = dnas_config.replace("NUM_JOBS_PER_GPU:1", f"NUM_JOBS_PER_GPU:{dnas_num_jobs_per_gpu}")
    sampled_config = sampled_config.replace("NUM_JOBS_PER_GPU:2", f"NUM_JOBS_PER_GPU:{sampled_num_jobs_per_gpu}")

    # Replace DNAS training arguments.
    for old_arg, new_arg in dnas_arg_changes.items():
        dnas_config = dnas_config.replace(old_arg, new_arg)

    # Replace sampled training arguments.
    for old_arg, new_arg in sampled_arg_changes.items():
        sampled_config = sampled_config.replace(old_arg, new_arg)

    # Write modified config strings back to new files.
    open(new_dnas_filename, "w").write(dnas_config)
    open(new_sampled_filename, "w").write(sampled_config)

    # Now, launch the run_kaggle_jobs.sh script.
    memory_map_str_arg = ("--memory_map" if memory_map else "no_mem_map")
    load_processed_data_str_arg = ("--load_processed" if load_processed_data else "")
    launch_jobs_arg = f"nohup ./run_kaggle_jobs.sh {python_cmd} {memory_map_str_arg} {kaggle_raw_file} {new_dnas_filename} {new_sampled_filename} {config_suffix_experiment_id} {mem_map_weights_folder} {mem_map_arch_params_folder} {search_space_group} {load_processed_data_str_arg}"

    # Launch jobs.
    os.system(launch_jobs_arg)
    print(f"STARTING JOBS WITH COMMAND {launch_jobs_arg}.")
