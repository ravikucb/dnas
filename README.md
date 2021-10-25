Differentiable Neural Architecture Search (DNAS) for Ads Click-through Rate (CTR) Prediction
=================================================================================
*Copyright (c) Facebook, Inc. and its affiliates.*

Description
--------------
An implementation of a Differentiable Neural Architecture Search (DNAS) framework, applied
to ads click-through rate (CTR) prediction. Contains implementations of search spaces using
the Deep Learning Recommendation Model (DLRM) as a backbone. The framework itself can be used
for any other NAS application, and the code in sample_extension/ shows an example of how this
would be done for search over CNNs for image classification.

Cite Work:

ADD ARXIV ARTICLE

Related Work:

For the original DNAS algorithm, applied to CNNs in [FBNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.pdf):

```
@inproceedings{wu2019fbnet,
  title       = {Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search},
  author      = {Wu, Bichen and Dai, Xiaoliang and Zhang, Peizhao and Wang, Yanghan and Sun, Fei and Wu, Yiming and Tian, Yuandong and Vajda, Peter and Jia, Yangqing and Keutzer, Kurt},
  booktitle   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages       = {10734--10742},
  year        = {2019}
}
```

For the original work on the [DLRM](https://arxiv.org/abs/1906.00091):

```
@article{DLRM19,
  author    = {Maxim Naumov and Dheevatsa Mudigere and Hao{-}Jun Michael Shi and Jianyu Huang and Narayanan Sundaraman and Jongsoo Park and Xiaodong Wang and Udit Gupta and Carole{-}Jean Wu and Alisson G. Azzolini and Dmytro Dzhulgakov and Andrey Mallevich and Ilia Cherniavskii and Yinghai Lu and Raghuraman Krishnamoorthi and Ansha Yu and Volodymyr Kondratenko and Stephanie Pereira and Xianjie Chen and Wenlin Chen and Vijay Rao and Bill Jia and Liang Xiong and Misha Smelyanskiy},
  title     = {Deep Learning Recommendation Model for Personalization and Recommendation Systems},
  journal   = {CoRR},
  volume    = {abs/1906.00091},
  year      = {2019},
  url       = {https://arxiv.org/abs/1906.00091},
}
```

Implementation
--------------
Please refer to the paper for detailed information regarding the implementation; however, we provide a brief summary of some major components below.

**nas_supernet.py**: Defines the SuperNet class that provides required functinality for any actual supernets.

**nas_searchmanager.py**: Implements the functionality to train a DNAS supernet and sample architectures from it.

**train_dnas.py**: Script to take arguments and train a CTR prediction supernet.

**train_sampled.py**: Script to take arguments and train a CTR prediction sampled architecture.

**run_dnas_pipeline.py**: Starts the supernet tuning and sampled architecture tuning jobs i.e. manages the entire DNAS pipeline.

**tuning.py**: Manages the launching of a number of jobs for hyperparameter (or other) tuning on a multi-GPU machine.

**dlrm_supernet.py**: Supernet for all DLRM CTR prediction experiments, which can run MLP search, embedding dimension search, or embedding cardinality search.

**test_dlrm_supernet.py**: Can test functionality of dlrm_supernet.py for different search spaces. However, this file should not be used to verify the implementation. It it more useful for verifying dlrm_supernet.py still works after any changes.

**nas_mlp.py**: Implements a supernet for MLP search.

**nas_embedding_dim.py**: Implements a supernet for embedding dimension search.

**nas_embedding_card.py**: Implements a supernet for embedding cardinality search.

**dnas_data_utils.py**: Utilities for data processing related to DNAS for Ads CTR prediction code.

**run_kaggle_jobs.sh**: Specialized script which specifies many arguments directly to run_dnas_pipeline.py and thus makes it easier to launch run_dnas_pipeline.py becuase the user only needs to specify a few arguments.

**dnas.py**: Provides a Python interface to launch a DNAS pipeline easily from a Python script. Using paper experiments as defaults for 3 search space groups already implemented, allows for modification of configurations.

**tuning_configs/**: This folder contains the tuning configurations used in our experiments.

**sample_extension/**: This folder contains a sample extension of the DNAS codebase, demonstrating how it might be used for an entirely different problem (CNN search for image classification).

Note that data_loader_terabyte.py, data_utils.py, dlrm_data_utils.py, and tricks/ are taken directly from the DLRM repo. dlrm_s_pytorch.py is slightly modified from the DLRM repo to allow for DLRMs with different dimensions for each sparse feature (used for embedding dimension search space).

How to run DNAS code?
--------------------
Before any of the Criteo Kaggle experiments can be run, you must [download](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) the data and put the resulting train.txt file in a folder called datasets/.

1) The search and hyperparameter tuning spaces originally used in the paper may be run with the following commands. Once one command has been run, passing the --load_processed argument to run_kaggle_jobs.sh (the last argument) to re-load the data in the datasets/ folder.

MLP search:
```
$ ./run_kaggle_jobs.sh python --memory_map datasets/train.txt tuning_configs/mlp_search_train_dnas_tuning_config tuning_configs/mlp_search_train_sampled_tuning_config mlp_search_criteo_kaggle datasets/weights_kaggle_train_full/split datasets/arch_params_kaggle_train_full/split top_bottom_mlps &
```

Embedding dimension search:
```
$ ./run_kaggle_jobs.sh python --memory_map datasets/train.txt tuning_configs/embedding_dimension_train_dnas_tuning_config tuning_configs/embedding_dimension_train_sampled_tuning_config embedding_dimension_search_criteo_kaggle datasets/weights_kaggle_train_full/split datasets/arch_params_kaggle_train_full/split emb_dim &
```

Embedding cardinality search:
```
$ ./run_kaggle_jobs.sh python --memory_map datasets/train.txt tuning_configs/embedding_cardinality_train_dnas_tuning_config tuning_configs/embedding_cardinality_train_sampled_tuning_config emb_card_search_criteo_kaggle datasets/weights_kaggle_train_full/split datasets/arch_params_kaggle_train_full/split emb_card &
```

We have also implemented a Python API in ```dnas.py``` which allows for launching the entire DNAS pipeline with a single import and Python command. This interface uses the experiments in the paper for each search space group as a set of default arguments and allows for changes to these default arguments to make experimentation easy. Note that this API is for experimentation with the 3 search space groups already implemented on the Criteo Kaggle dataset; while it should be reasonably straightforward to extend to other search space groups or datasets (as with the entire codebase), this is not currently supported.

One can launch experiments for the top_bottom_mlps, emb_dim, or emb_card search space groups. Please refer to ```dnas.py``` for more detailed information, but the main way in which changes are made to the default configurations is through a dictionary containing such changes to be made, which ```run_pipeline``` makes, putting the results in a new set of config files. Note that what is given to the ```run_pipeline``` function and what is contained in the tuning configuration files (after modification by ```run_pipeline```) should always match, and this is especially importnat in the case of data processing arguments. The below command is an example of what might be run for a modified emb_card search space:

```
from dnas import run_pipeline
run_pipeline(search_space_group="emb_card", config_suffix_experiment_id="modified_emb_card", train_dnas_gpus="0,2,3", train_sampled_gpus="0,2,3", dnas_num_jobs_per_gpu=2, sampled_num_jobs_per_gpu=2, dnas_arg_changes={"--cardinality_options 1.0 0.1 0.01 0.001" : "--cardinality_options 1.0 0.1 0.01 0.001 0.0001"}, sampled_arg_changes={"--lr={0.25, 0.5, 1.0, 2.0}" : "--lr={0.25, 0.5, 1.0}"}, python_cmd="python", memory_map=True, kaggle_raw_file="datasets/train.txt", mem_map_weights_folder="datasets/weights_kaggle_train_full/split", mem_map_arch_params_folder="datasets/arch_params_kaggle_train_full/split", load_processed_data=False)
```

2) A sample extension of the code, which runs DNAS over a CNN, can be called with the following commands.
```
$ cd sample_extension
$ python run_dnas_pipeline.py --python_cmd=python --train_dnas_tuning_config=config_cnn_dnas_search --train_sampled_tuning_config=config_cnn_sampled_search --global_experiment_id=dnas_cnn_test_tuning &
```

3) After running the sample extension, the results can be verified by comparing them with the expected_output in the sample_extension/ folder. The verify_output.py script automates this process:
```
$ python verify_output.py --output_to_verify all_configs_no_checkpoints_save_file_dnas_cnn_test_tuning_sampled
```

Version
-------
0.1 : Initial release of the DNAS for ads code

Requirements
------------
pytorch (1.4.0)

numpy

tqdm

onnx

sklearn

License
-------
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
