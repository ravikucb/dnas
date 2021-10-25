Sample Extension to DNAS for Ads CTR Prediction
=================================================================================
*Copyright (c) Facebook, Inc. and its affiliates.*

The code in this folder is a self-contained example extension to the DNAS codebase, intended to help users adapt our framework for their own applications.

Specifically, this sample implements a search space over convolutional networks for image classification on a synthetic dataset.

Please note that the network structure is NOT designed to be practically useful, and is instead designed to be a simple example of how to extend our implementation of DNAS.

ARCHITECTURE SEARCH SPACE:

INPUT IMAGE 3x50x50 --> LAYER CHOICE 32x (3x3, 5x5, 7x7) = 32x50x50 --> 2x2 MAX POOL 32x25x25 --> LAYER CHOICE 64x (3x3, 5x5, 7x7) = 64x25x25 --> 5x5 MAX POOL 64x5x5 --> FLATTEN 1600 --> CLASSIFICATION (SOFTMAX) 5

The tuning for the supernets and sampled architectures is done only over learning rates.

The entire pipeline can be started with the command:

```
$ python run_dnas_pipeline.py --python_cmd=python --train_dnas_tuning_config=config_cnn_dnas_search --train_sampled_tuning_config=config_cnn_sampled_search --global_experiment_id=dnas_cnn_test_tuning &
```

The DNAS installation can be verified using this sample extension by running the following command (after the DNAS pipeline command above is run of course):

```
$ python verify_output.py --output_to_verify all_configs_no_checkpoints_save_file_dnas_cnn_test_tuning_sampled
```
