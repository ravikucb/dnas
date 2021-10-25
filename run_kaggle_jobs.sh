# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/25/21

# Python command is the first argument, then memory map.
PYTHON_CMD=$1
MEMORY_MAP_ARG=$2

# Assumes that train.txt file is in datasets folder.
KAGGLE_RAW_FILE=$3

# Tuning configuration files.
DNAS_CONFIG=$4
SAMPLED_CONFIG=$5

# Experiment ID.
EXPERIMENT_ID=$6

# Specific to memory map.
MEM_MAP_WEIGHTS_FOLDER=$7
MEM_MAP_ARCH_PARAMS_FOLDER=$8

# Search space arg.
SEARCH_SPACE=$9

# Load processed data.
PROCESSED_DATA_ARG=${10}

# Starts run_dnas_pipeline.py depending on memory map argument. For the emb_card search space,
# anything passed to --embeddings_num_vectors will be ignored and no categories will be dropped
# in the initial data generation.
if [ "$MEMORY_MAP_ARG" = "no_mem_map" ] ; then
    nohup $PYTHON_CMD run_dnas_pipeline.py --python_cmd=$PYTHON_CMD --search_space=$SEARCH_SPACE --embeddings_num_vectors 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 --dataset=kaggle --raw_data_file=$KAGGLE_RAW_FILE --processed_data_file=kaggleAdDisplayChallenge_processed $PROCESSED_DATA_ARG --data_randomization=total --category_dropping=modulo --train_dnas_tuning_config=$DNAS_CONFIG --train_sampled_tuning_config=$SAMPLED_CONFIG --global_experiment_id=$EXPERIMENT_ID &
    echo "JOBS WILL BE STARTED WITHOUT --memory_map"
else
    nohup $PYTHON_CMD run_dnas_pipeline.py --python_cmd=$PYTHON_CMD --search_space=$SEARCH_SPACE --embeddings_num_vectors 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 10000 --dataset=kaggle --raw_data_file=$KAGGLE_RAW_FILE --processed_data_file=kaggleAdDisplayChallenge_processed $PROCESSED_DATA_ARG --mem_map_weights_data_file=$MEM_MAP_WEIGHTS_FOLDER --mem_map_arch_params_data_file=$MEM_MAP_ARCH_PARAMS_FOLDER --memory_map --data_randomization=total --category_dropping=modulo --train_dnas_tuning_config=$DNAS_CONFIG --train_sampled_tuning_config=$SAMPLED_CONFIG --global_experiment_id=$EXPERIMENT_ID &
    echo "JOBS WILL BE STARTED WITH --memory_map"
fi
