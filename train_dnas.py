#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Ravi Krishna 07/25/21

# Various import statements.
import torch
import torch.nn as nn
from nas_searchmanager import SearchManager
from dlrm_supernet import DLRMSuperNet
import dlrm_data_pytorch as dp
import argparse
from dnas_data_utils import get_dnas_dataloaders
import pickle
from utils import arch_sampling_str_to_dict, STR_TO_OPTIM
import random
import numpy as np
import time
import os

# Create argument parser.
parser = argparse.ArgumentParser(description="Run DNAS DLRM supernet training.")

# Supernet parameters.
parser.add_argument("--search_space",
                    type=str,
                    choices=["top_bottom_mlps", "emb_dim", "emb_card"],
                    default="top_bottom_mlps",
                    help="What type of search space DNAS should use.")
parser.add_argument("--embedding_dimension",
                    type=int,
                    default=None,
                    help="Supernet embedding dimension.")
parser.add_argument("--dimension_options",
                    type=int,
                    nargs="+",
                    help="Dimension options if embedding dimension search space is being used.")
parser.add_argument("--cardinality_options",
                    type=float,
                    nargs="+",
                    help="Embedding cardinality options expressed as a proportion of total categories kept.")
parser.add_argument("--feature_counts_file",
                    type=str,
                    default=None,
                    help="Feature counts file; must be specific for emb_card search space.")
parser.add_argument("--embeddings_num_vectors",
                    type=int,
                    nargs="+",
                    help="Number of vectors in each embedding table e.g. \"10000 1000 100\" for a dataset with 3 categorical features. Note that this should include a category for missing categorical feature data.")
parser.add_argument("--num_dense_features",
                    type=int,
                    default=None,
                    help="Number of dense features in supernet input.")
parser.add_argument("--bottom_mlp_layer_options",
                    type=int,
                    nargs="+",
                    help="Possible FC sizes for bottom MLP.")
parser.add_argument("--max_bottom_mlp_layers",
                    type=int,
                    default=None,
                    help="Max number of bottom MLP layers.")
parser.add_argument("--top_mlp_layer_options",
                    type=int,
                    nargs="+",
                    help="Possible FC sizes for top MLP.")
parser.add_argument("--max_top_mlp_layers",
                    type=int,
                    default=None,
                    help="Max number of top MLP layers.")
parser.add_argument("--op_list_path",
                    type=str,
                    default=None,
                    help="Path to list of possible FC ops and corresponding cost LUT index. Both --op_list_path and --cost_lut_path must be specified to use the imported LUT and not the one automatically generated.")
parser.add_argument("--cost_lut_path",
                    type=str,
                    default=None,
                    help="Path to HW cost LUT. Both --op_list_path and --cost_lut_path must be specified to use the imported LUT and not the one automatically generated.")
parser.add_argument("--arch_interaction_op",
                    type=str,
                    choices=['dot', 'cat'],
                    default="dot",
                    help="Interaction operator; must be one of dot or cat.")
parser.add_argument("--arch_include_self_interaction",
                    action="store_true",
                    help="Whether or not to include self interactions in dot product feature interactions layer.")
parser.add_argument("--loss_threshold",
                    type=float,
                    default=1e-9,
                    help="Loss threshold to prevent BCELoss errors; predicted probabilities clamped between loss_threshold and (1 - loss_threshold).")

# Dataset parameters.
parser.add_argument("--dataset",
                    type=str,
                    choices=["kaggle"],
                    default="kaggle",
                    help="Dataset to use; currently only kaggle is supported.")
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

# Training / search manager parameters.
parser.add_argument("--experiment_id",
                    type=str,
                    default=None,
                    help="Unique experiment ID used as a prefix for all files saved during the experiment, including sampled architectures and logfiles.")
parser.add_argument("--weights_batch_size",
                    type=int,
                    default=None,
                    help="Weights training batch size.")
parser.add_argument("--arch_params_batch_size",
                    type=int,
                    default=None,
                    help="Arch params training batch size.")
parser.add_argument("--initial_temperature",
                    type=float,
                    default=None,
                    help="Initial Gumbel Softmax temperature.")
parser.add_argument("--temp_decay_rate",
                    type=float,
                    default=None,
                    help="Decay rate of Gumbel Softmax temperature.")
parser.add_argument("--architecture_sampling",
                    type=str,
                    default="",
                    help="Architecture sampling. To sample 4 architecture after 1 epoch of architecture parameters training, 4 after 2, etc. for all 4 epochs, one would write \"1:4,2:4,3:4,4:4\".")
parser.add_argument("--n_warmup_epochs",
                    type=float,
                    default=None,
                    help="Number (possibly float) of warmup epochs i.e. weights-only training before architecture parameters trained.")
parser.add_argument("--n_total_s_net_training_epochs",
                    type=float,
                    default=None,
                    help="Total (possibly float) number of supernet training epochs.")
parser.add_argument("--n_alt_train_epochs",
                    type=float,
                    default=None,
                    help="Every n_alt_train_epochs, we switch from training the weights to architecture parameters or vice versa.")
parser.add_argument("--host_gpu_id",
                    type=int,
                    default=None,
                    help="Host GPU ID.")
parser.add_argument("--clip_grad_norm_value",
                    type=float,
                    default=None,
                    help="L2 norm at which to clip gradients of supernet.")    # Both weights and architecture parameters gradients.
parser.add_argument("--weights_optim_type",
                    type=str,
                    choices=["sgd"],
                    default="sgd",
                    help="Weights optimizer type.")    # DLRM code uses sparse embedding gradients, requiring either SparseAdam or SGD. SparseAdam does not support the dense gradients of the supernet, so SGD is currently the only feasible option.
parser.add_argument("--arch_params_optim_type",
                    type=str,
                    choices=["sgd", "adam", "adagrad"],
                    default="sgd",
                    help="Architecture parameters optimizer type.")
parser.add_argument("--weights_lr",
                    type=float,
                    default=None,
                    help="Initial learning rate for architecture weights.")
parser.add_argument("--arch_params_lr",
                    type=float,
                    default=None,
                    help="Initial learning rate for architecture configuration parameters.")
parser.add_argument("--weights_wd",
                    type=float,
                    default=None,
                    help="Weight decay for architecture weights.")
parser.add_argument("--arch_params_wd",
                    type=float,
                    default=None,
                    help="Weight decay for architecture configuration parameters.")
parser.add_argument("--use_hw_cost",
                    action="store_true",
                    help="Whether or not to use HW cost in the DNAS training.")
parser.add_argument("--load_cost_table_data_path",
                    type=str,
                    default="",
                    help="File from which to load cost table - should be a Python pickle file with {\"indices_look_up_table\": DICTIONARY TO LOOK UP COST TABLE INDICES FROM OPERATOR NAMES, \"cost_table\": PYTHON LIST OR NP.ARRAY OF HW COSTS (E.G. LATENCY)}")
parser.add_argument("--hw_cost_function",
                    type=str,
                    choices=["exponential", "linear"],
                    default="exponential",
                    help="HW cost function type if --use_hw_cost.")
parser.add_argument("--hw_cost_exp",
                    type=float,
                    default=None,
                    help="HW cost function exponent, provided only if --use_hw_cost and --hw_cost_function=exponential.")
parser.add_argument("--hw_cost_coef",
                    type=float,
                    default=None,
                    help="HW cost linear coefficient, provided if --use_hw_cost.")
parser.add_argument("--hw_cost_multiplier",
                    type=float,
                    default=1.0,
                    help="Linear HW cost multiplier to e.g. convert latency numbers measured in seconds to milliseconds.")
parser.add_argument("--weights_lr_base",
                    type=float,
                    default=None,
                    help="Weights LR = weights_lr * ((weights_lr_base) ** (num_weights_epochs)). Note that this formula may be applied at every training step or every n_alt_train_epochs.")
parser.add_argument("--arch_params_lr_base",
                    type=float,
                    default=None,
                    help="Arch params LR = arch_params_lr * ((arch_params_lr_base) ** (num_arch_params_epochs)). Note that this formula may be applied at every training step or every n_alt_train_epochs.")    # Every epoch not currenly an option - may be added later.
parser.add_argument("--update_lrs_every_step",
                    action="store_true",
                    help="If set, LRs will be updated every step instead of every SearchManager \"epoch\" (usually args.n_alt_train_amt).")    # Could update the weights and architecture parameters learning rates at different frequencies.

# Seed.
parser.add_argument("--seed",
                    type=int,
                    default=1,
                    help="Random seed to ensure results can be replicated. This seed is used for random, numpy, and torch.")

# Needed to interface with tuning script.
parser.add_argument("--save_metrics_param",
                    type=str,
                    default="",
                    help="Path at which to save a file to tell tuning.py that this script is done running.")

# Parse arguments.
args = parser.parse_args()

# Set seed.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Load the indices look up table and cost table if needed.
if args.use_hw_cost:
    if args.load_cost_table_data_path != "":
        loaded_cost_table_data = pickle.load(args.load_cost_table_data_path)

        indices_look_up_table = loaded_cost_table_data["indices_look_up_table"]
        cost_table = loaded_cost_table_data["cost_table"]

# If necessary, load feature counts directly.
if args.search_space == "emb_card":
    args.embeddings_num_vectors = np.load(args.feature_counts_file)["counts"]

# Create the DLRM supernet.
if args.search_space == "top_bottom_mlps" or args.search_space == "emb_dim" or args.search_space == "emb_card":
    dlrm_supernet = DLRMSuperNet(search_space=args.search_space,
                                emb_dim=(args.embedding_dimension if args.search_space == "top_bottom_mlps" or args.search_space == "emb_card" else args.dimension_options),
                                embs_n_vectors=args.embeddings_num_vectors,
                                n_dense_features=args.num_dense_features,
                                bottom_mlp_sizes=args.bottom_mlp_layer_options,
                                max_n_bottom_mlp_layers=args.max_bottom_mlp_layers,
                                top_mlp_sizes=args.top_mlp_layer_options,
                                max_n_top_mlp_layers=args.max_top_mlp_layers,
                                interaction_op=args.arch_interaction_op,
                                include_self_interaction=args.arch_include_self_interaction,
                                gen_cost_table=(True if (args.use_hw_cost and args.load_cost_table_data_path == "") else False),
                                indices_look_up_table=(indices_look_up_table if args.load_cost_table_data_path != "" else None),
                                cost_table=(cost_table if args.load_cost_table_data_path != "" else None),
                                loss_threshold=args.loss_threshold,
                                emb_card_options=args.cardinality_options,
                                enable_float_card_options=(args.search_space == "emb_card"))

# Get dataloaders. When starting many jobs at the same time, some seem to fail due to an EOFError.
dataloaders_loaded = False
while not dataloaders_loaded:
    try:
        weights_dataloader, arch_params_dataloader = get_dnas_dataloaders(args)
        dataloaders_loaded = True
    except Exception as dataloader_exception:
        print(f"ENCOUNTERED DATALOADER EXCEPTION IN train_dnas.py: {dataloader_exception}.")

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

    # Exit.
    exit()

# Move DLRM supernet to GPU.
try:
    host_device = torch.device(f"cuda:{args.host_gpu_id}")
    print(f"ATTEMPTING TO MOVE DLRM SUPERNET TO GPU {args.host_gpu_id}.")
    print(dlrm_supernet)
    dlrm_supernet.to(host_device)
except RuntimeError as oom_error:
    write_oom_exit(oom_error)

# Construct various inputs to SearchManager.__init__():

# Optimizer classes.
weights_optim_class = STR_TO_OPTIM[args.weights_optim_type.lower()]
arch_params_optim_class = STR_TO_OPTIM[args.arch_params_optim_type.lower()]

# Optimizer initialization parameters.
weights_optim_init_params = {"lr" : args.weights_lr, "weight_decay" : args.weights_wd}
arch_params_optim_init_params = {"lr" : args.arch_params_lr, "weight_decay" : args.arch_params_wd}

# Functions to fetch parameters that each optimizer should train.
weights_parameters_function = lambda s_net: (list(s_net.bot_l.parameters()) + list(s_net.top_l.parameters()) + list(s_net.emb_l.parameters()))

if args.search_space == "top_bottom_mlps":
    arch_params_parameters_function = lambda s_net: (list(s_net.bot_l.theta_parameters.parameters()) + list(s_net.top_l.theta_parameters.parameters()))
elif args.search_space == "emb_dim" or args.search_space == "emb_card":
    def arch_params_parameters_function(s_net):
        theta_parameters = []
        for emb_ix in range(len(s_net.embs_n_vectors)):
            theta_parameters += s_net.emb_l[emb_ix].theta_parameters.parameters()
        return theta_parameters

# Functions which specify how the LR changes during training. Note that
# these functions return the RATIO of the current learning rate to the
# initial learning rate, and not the current learning rate itself.
weights_optim_lr_lambdas = [lambda curr_epoch: (args.weights_lr_base ** curr_epoch)]
arch_params_optim_lr_lambdas = [lambda curr_epoch: (args.arch_params_lr_base ** curr_epoch)]

# Initial learning rates for the different parameters groups in each
# optimizer. Currently there is only one parameter group used per optimizer,
# however, the code supports multiple parameters groups, each with their own
# initial learning rate and learning rate schedule.
weights_initial_lrs = [args.weights_lr]
arch_params_initial_lrs = [args.arch_params_lr]

# The only loss function currently supported is binary cross-entropy (BCE).
# Note: instantiating the loss function here shouldn't cause any problems,
# but in case it does, we can just pass the nn.BCELoss class and then
# instantiate it in SearchManager.train_dnas().
loss_function = nn.BCELoss()

# Create search_manager.
search_manager = SearchManager(super_net=dlrm_supernet,
                                init_temp=args.initial_temperature,
                                temp_decay_rate=args.temp_decay_rate,
                                n_warmup_epochs=args.n_warmup_epochs,
                                arch_sampling=arch_sampling_str_to_dict(args.architecture_sampling),
                                n_total_s_net_train_epochs=args.n_total_s_net_training_epochs,
                                n_alt_train_amt=args.n_alt_train_epochs,
                                host_device=host_device,
                                clip_grad_norm_value=args.clip_grad_norm_value,
                                w_dataloader=weights_dataloader,
                                m_dataloader=arch_params_dataloader,
                                w_optim_class=weights_optim_class,
                                weights_optim_init_params=weights_optim_init_params,
                                w_optim_params_func=weights_parameters_function,
                                m_optim_class=arch_params_optim_class,
                                mask_optim_init_params=arch_params_optim_init_params,
                                m_optim_params_func=arch_params_parameters_function,
                                weights_lr_lambdas=weights_optim_lr_lambdas,
                                mask_lr_lambdas=arch_params_optim_lr_lambdas,
                                weights_initial_lrs=weights_initial_lrs,
                                mask_initial_lrs=arch_params_initial_lrs,
                                update_lrs_every_step=args.update_lrs_every_step,
                                loss_function=loss_function,
                                experiment_id=args.experiment_id,
                                logfile=args.experiment_id.replace("save_file", "search_manager_logfile"),
                                use_hw_cost=args.use_hw_cost,
                                cost_exp=args.hw_cost_exp,
                                cost_coef=args.hw_cost_coef,
                                exponential_cost=(True if args.use_hw_cost and args.hw_cost_function == "exponential" else False),
                                cost_multiplier=args.hw_cost_multiplier)

# Start search process.
try:
    search_manager.train_dnas()
except RuntimeError as oom_error:
    write_oom_exit(oom_error)

# Once the DNAS process is done, in order to tuning.py to know
# that the script is done running, save a file at the save_metrics_param
# location.
with open(args.save_metrics_param, "wb") as save_metrics_writefile:
    pickle.dump({"info" : "SCRIPT COMPLETED"}, save_metrics_writefile)
