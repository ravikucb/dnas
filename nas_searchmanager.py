#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Ravi Krishna 07/25/21

# Note that the code to use the performance model
# (i.e. multiplying it with the loss, taking a *
# (cost ** b)) is from Bichen Wu's search manager
# code for FBNet.

# The idea of the super net returning the model
# outputs as well as the cost is from
# Bichen Wu's FBNet code.

# Various import statements.
import torch
import torch.nn as nn
import numpy as np
import pickle
from utils import step_lambda_lr
import time
import os
from torch.nn.utils import clip_grad_norm_ as clip_grad

class SearchManager(object):
    def __init__(self,
                super_net=None,
                init_temp=None,
                temp_decay_rate=None,
                n_warmup_epochs=None,
                arch_sampling=None,
                n_total_s_net_train_epochs=None,
                n_alt_train_amt=None,
                host_device=None,
                clip_grad_norm_value=None,
                w_dataloader=None,
                m_dataloader=None,
                w_optim_class=None,
                weights_optim_init_params=None,
                w_optim_params_func=None,
                m_optim_class=None,
                mask_optim_init_params=None,
                m_optim_params_func=None,
                weights_lr_lambdas=None,
                mask_lr_lambdas=None,
                weights_initial_lrs=None,
                mask_initial_lrs=None,
                update_lrs_every_step=None,
                loss_function=None,
                experiment_id="",
                logfile="",
                use_hw_cost=True,
                cost_exp=None,
                cost_coef=None,
                exponential_cost=True,
                cost_multiplier=None):

        """
        Initializes the SearchManager object.

        SearchManager manages the DNAS search process. The outputs
        that SearchManager produces are primarily saved architecutre
        configurations which are hard-sampled from the Gumbel Softmax
        distribution at arbitrary points during supernet training.

        These architecture samples are then trained from scratch
        later on.
        """

        # Read various parameters from parameters and store for use later.
        self.super_net = super_net

        self.init_temp = init_temp
        self.temp_decay_rate = temp_decay_rate
        self.num_warmup_epochs = n_warmup_epochs
        self.architecture_sampling = arch_sampling
        self.n_total_s_net_train_epochs = n_total_s_net_train_epochs
        self.n_alt_train_amt = n_alt_train_amt
        self.host_device = host_device
        self.clip_grad_norm_value = clip_grad_norm_value

        self.w_dataloader = w_dataloader
        self.m_dataloader = m_dataloader
        self.w_optim_class = w_optim_class
        self.weights_optim_init_params = weights_optim_init_params
        self.w_optim_params_func = w_optim_params_func
        self.m_optim_class = m_optim_class
        self.mask_optim_init_params = mask_optim_init_params
        self.m_optim_params_func = m_optim_params_func
        self.weights_lr_lambdas = weights_lr_lambdas
        self.mask_lr_lambdas = mask_lr_lambdas
        self.weights_initial_lrs = weights_initial_lrs
        self.mask_initial_lrs = mask_initial_lrs
        self.update_lrs_every_step = update_lrs_every_step

        self.loss_function = loss_function

        self.experiment_id = experiment_id
        self.logfile_name = logfile

        # Parameters to use the HW cost.

        # If this is set to False, then the DNAS
        # search will optimize only for the task
        # loss and ignore the HW cost term.
        self.use_hw_cost = use_hw_cost

        self.cost_exp = cost_exp
        self.cost_coef = cost_coef
        self.exponential_cost = exponential_cost

        # Linear cost multiplier; useful if
        # the cost is measured in seconds,
        # but we want to convert it to
        # milliseconds or microseconds so
        # that the weighting of the task and
        # HW cost is different.
        self.cost_multiplier = cost_multiplier

        # Number of architectures already sampled.
        self.sampled_arch_ix = 0

    def calc_epoch_training_params(self,
                                alt_train_period,
                                num_warmup_epochs,
                                total_num_training_epochs,
                                init_temp,
                                temp_decay_rate,
                                arch_sampling_dict):
        """
        Returns information about each epoch
        to be completed; note that each epoch
        that we refer to here may not actually
        be a full training epoch in terms of
        going through every example but rather
        is alt_train_period epochs
        of training for some or all parameters.
        """

        def get_curr_temp(current_epoch,
                        init_temp,
                        temp_decay_rate):
            """
            This function just computes the
            Gumbel Softmax temperature using
            the exponential decay formula.
            """

            curr_exponent = -temp_decay_rate * current_epoch
            return init_temp * (np.e ** (curr_exponent))

        all_epochs = []
        curr_epochs_done = 0.00
        while curr_epochs_done < total_num_training_epochs:
            if curr_epochs_done < num_warmup_epochs:
                curr_temp = get_curr_temp(0.00, init_temp, temp_decay_rate)
                all_epochs.append({"weights_start" : curr_epochs_done,
                                "mask_start" : 0.00,
                                "what_to_train" : "weights",
                                "weights_end" : curr_epochs_done + alt_train_period,
                                "mask_end" : 0.00,
                                "epoch_type" : "warmup",
                                "temperature" : curr_temp,
                                "architectures_to_sample" : 0})
            else:
                # Check if we need to sample architecture
                # at the end. We sample them after
                # training the architecture parameters.

                n_archs_to_sample = 0
                try:
                    # The number of mask epochs that
                    # will be completed at the end
                    # of this epoch is
                    # curr_epochs_done +
                    # alt_train_period.
                    total_epochs_samp = curr_epochs_done + alt_train_period
                    samp_dict_n = total_epochs_samp - num_warmup_epochs
                    n_archs_to_sample = arch_sampling_dict[samp_dict_n]
                except KeyError:
                    pass

                # Add training epochs for
                # both weights and mask.

                m_start_epochs = curr_epochs_done - num_warmup_epochs
                w_end_epochs = curr_epochs_done + alt_train_period
                m_end_epochs = curr_epochs_done - num_warmup_epochs

                # Current temperature.
                curr_temp = get_curr_temp(m_start_epochs, init_temp, temp_decay_rate)

                all_epochs.append({"weights_start" : curr_epochs_done,
                                        "mask_start" : m_start_epochs,
                                        "what_to_train" : "weights",
                                        "weights_end" : w_end_epochs,
                                        "mask_end" : m_end_epochs,
                                        "epoch_type" : "weights_training",
                                        "temperature" : curr_temp,
                                        "architectures_to_sample" : 0})

                w_start_epochs = w_end_epochs
                m_end_epochs = m_end_epochs + alt_train_period
                all_epochs.append({"weights_start" : w_start_epochs,
                                        "mask_start" : m_start_epochs,
                                        "what_to_train" : "mask",
                                        "weights_end" : w_end_epochs,
                                        "mask_end" : m_end_epochs,
                                        "epoch_type" : "mask_training",
                                        "temperature" : curr_temp,
                                        "architectures_to_sample" : n_archs_to_sample})

            curr_epochs_done += alt_train_period

        return all_epochs

    # Utility function.
    # Very slightly modified from dlrm_s_pytorch.py.
    def move_data_to_gpu(self, X, lS_o, lS_i, T, device):
        # lS_i can be either a list of tensors or a stacked tensor.
        # Handle each case below:
        lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
            else lS_i.to(device)
        lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
            else lS_o.to(device)

        return X.to(device), \
                lS_o, \
                lS_i, \
                T.to(device)

    def run_one_dnas_step(self, current_epoch, batch_idx, curr_temp, X, lS_o, lS_i, T,
            weights_optimizer, mask_optimizer):
        # Move the data and target
        # tensors to self.host_device.
        dense_features, sparse_offsets, sparse_indices, labels = self.move_data_to_gpu(X, lS_o, lS_i, T, self.host_device)

        # Zero gradients.
        weights_optimizer.zero_grad()
        mask_optimizer.zero_grad()

        # Run the model forward pass.
        super_net_outputs = self.super_net(dense_features,
                                sparse_offsets,
                                sparse_indices,
                                sampling="soft",
                                temperature=curr_temp)

        # If self.use_hw_cost is True then the
        # super net should also return the
        # model cost; if not, it should just
        # return the model outputs.
        if self.use_hw_cost is False:
            model_predictions = super_net_outputs

        elif self.use_hw_cost is True:
            model_predictions, model_cost = super_net_outputs

        # Calculate the loss function.
        loss = self.loss_function(model_predictions, labels)
        if self.use_hw_cost is True:
            correct_time_model_cost = model_cost * self.cost_multiplier

            # Use either the exponential or linear cost function.
            if self.exponential_cost:
                loss = loss * ((correct_time_model_cost.log() ** self.cost_exp).mean()) * self.cost_coef
            else:
                mean_log_cost = correct_time_model_cost.log().mean()
                loss = loss + (self.cost_coef * mean_log_cost / 100.0)

        # Backpropagation.
        loss.backward()
        clip_grad(self.super_net.parameters(), self.clip_grad_norm_value)

        # Run the correct optimizer.
        if current_epoch["what_to_train"] == "weights":
            weights_optimizer.step()

        elif current_epoch["what_to_train"] == "mask":
            mask_optimizer.step()

        # Return the loss value.
        return float(loss.item())

    def sample_archs(self, current_epoch):
        saved_arch_fnames_local = []

        # If there are no architectures to sample,
        # then this loop will just never run which is fine.
        for curr_sampled_arch_ix in range(current_epoch["architectures_to_sample"]):
            print("SAMPLING AN ARCHITECTURE:", current_epoch)

            # Sample architecture and save.
            sampled_arch_config = self.super_net.sample_arch()
            saved_architecture_filename = f"{self.experiment_id}_sampled_arch_{self.sampled_arch_ix}"

            with open(saved_architecture_filename, "wb") as writefile:
                # Get theta params as list.
                list_theta_params = []
                for curr_param in self.super_net.theta_parameters:
                    list_theta_params.append(curr_param.detach().cpu().numpy())

                # Create keys and values.
                keys = ["arch_config",
                        "theta_parameters",
                        "weights_epochs_trained_before_sampling",
                        "mask_epochs_trained_before_sampling",
                        "local_sampling_index"]

                values = [sampled_arch_config,
                            list_theta_params,
                            current_epoch["weights_end"],
                            current_epoch["mask_end"],
                            curr_sampled_arch_ix]

                # Create dict.
                save_dict = {key : value for key, value in zip(keys, values)}

                # Save dict. Note that this dictionary contains NO TENSORS and so is saved with pickle.
                pickle.dump(save_dict, writefile)

            # Store the saved architecture filenames in the local list.
            saved_arch_fnames_local.append(saved_architecture_filename)

            # Increment the sampled architecture counter.
            self.sampled_arch_ix += 1

        # Return all architectures saved in this function.
        return saved_arch_fnames_local

    def train_dnas(self):
        """
        Runs the overall training process
        for Differentiable Neural
        Architecture Search (DNAS).
        """

        # Record the start time.
        dnas_start_time = time.time()

        # Initialize the optimizers etc.
        weights_optimizer = self.w_optim_class(self.w_optim_params_func(self.super_net),
                                                **self.weights_optim_init_params)
        mask_optimizer = self.m_optim_class(self.m_optim_params_func(self.super_net),
                                                **self.mask_optim_init_params)

        # Get all of the epochs to be completed.
        all_epochs = self.calc_epoch_training_params(self.n_alt_train_amt,
                                                        self.num_warmup_epochs,
                                                        self.n_total_s_net_train_epochs,
                                                        self.init_temp,
                                                        self.temp_decay_rate,
                                                        self.architecture_sampling)

        # Calculate the number of steps in the
        # weights and validation dataloaders.
        w_batch_size = self.w_dataloader.batch_size
        w_dataset_len = len(self.w_dataloader.dataset)
        m_batch_size = self.m_dataloader.batch_size
        m_dataset_len = len(self.m_dataloader.dataset)
        n_w_batches = int(1.0 + (float(w_dataset_len) / float(w_batch_size)))
        n_m_batches = int(1.0 + (float(m_dataset_len) / float(m_batch_size)))

        # Store the saved architectures for later.
        saved_arch_fnames = []

        # Set the model to training mode.
        self.super_net.train()

        # Record all of the loss values.
        loss_values = []

        # Iterate through all of the epochs.
        for current_epoch in all_epochs:
            print(current_epoch)
            with open(self.logfile_name, "a") as logfile_open:
                logfile_open.write(str(current_epoch) + "\n")
                logfile_open.flush()

            # Get the current dataloader
            # (and the number of steps
            # in it) and temperature, as
            # well as the number of steps
            # to skip at the beginning of
            # the epoch and the total
            # number of steps to train.
            if current_epoch["what_to_train"] == "mask":
                current_dataloader = self.m_dataloader
            else:
                current_dataloader = self.w_dataloader

            # Steps in dataloader depending
            # on what we are training.
            if current_epoch["what_to_train"] == "mask":
                n_batches = n_m_batches
            else:
                n_batches = n_w_batches

            # Current temperature.
            curr_temp = current_epoch["temperature"]

            # We currently calculate the
            # start and end epochs based
            # on what we are training;
            # this will maintain
            # continuous epochs for both
            # the weights and the mask.

            # Number of epochs already done
            # and number that will be done
            # once this epoch or partial
            # epoch is completed.
            if current_epoch["what_to_train"] == "weights":
                epochs_so_far = current_epoch["weights_start"]
                epochs_once_done = current_epoch["weights_end"]
            else:
                epochs_so_far = current_epoch["mask_start"]
                epochs_once_done = current_epoch["mask_end"]

            # batches probably better than steps for name.
            frac_epoch = float(float(epochs_so_far) - float(int(epochs_so_far)))
            steps_to_skip = int(frac_epoch * float(n_batches))

            to_do_epoch = float(float(epochs_once_done) - float(epochs_so_far))
            steps_to_train = int(to_do_epoch * float(n_batches))

            # Adjust the learning rates; run
            # step_lambda_lr for both the
            # weights and the mask optimizers.
            weights_epochs_completed = current_epoch["weights_start"]
            step_lambda_lr(weights_optimizer,
                            self.weights_lr_lambdas,
                            weights_epochs_completed,
                            self.weights_initial_lrs)

            mask_epochs_completed = current_epoch["mask_start"]
            step_lambda_lr(mask_optimizer,
                            self.mask_lr_lambdas,
                            mask_epochs_completed,
                            self.mask_initial_lrs)

            # Store the number of steps trained in the epoch.
            steps_trained = 0

            # Training loop.
            for batch_idx, (X, lS_o, lS_i, T) in enumerate(current_dataloader):
                # Check if we need to skip this
                # training step; if so, skip it.
                if batch_idx < steps_to_skip:
                    continue

                # If this is the last batch with incorrec batch size, skip.
                if list(T.size())[0] != current_dataloader.batch_size:
                    continue

                # Run one training step.
                curr_loss_value = self.run_one_dnas_step(current_epoch, batch_idx,
                    curr_temp, X, lS_o, lS_i, T, weights_optimizer, mask_optimizer)
                loss_values.append(curr_loss_value)

                # Increment the trained steps counter.
                steps_trained += 1

                # Check if we need to update the learning rates.
                if self.update_lrs_every_step:
                    if current_epoch["what_to_train"] == "weights" and current_epoch["epoch_type"] == "warmup":
                         epoch_length = current_epoch["weights_end"] - current_epoch["weights_start"]
                         weights_epochs_completed = current_epoch["weights_start"] + ((steps_trained / steps_to_train) * epoch_length)
                         step_lambda_lr(weights_optimizer,
                                        self.weights_lr_lambdas,
                                        weights_epochs_completed,
                                        self.weights_initial_lrs)

                    elif current_epoch["what_to_train"] == "weights" and current_epoch["epoch_type"] == "weights_training":
                         epoch_length = current_epoch["weights_end"] - current_epoch["weights_start"]
                         weights_epochs_completed = current_epoch["weights_start"] + ((steps_trained / steps_to_train) * self.n_alt_train_amt)
                         step_lambda_lr(weights_optimizer,
                                        self.weights_lr_lambdas,
                                        weights_epochs_completed,
                                        self.weights_initial_lrs)

                    elif current_epoch["what_to_train"] == "mask":
                         epoch_length = current_epoch["mask_end"] - current_epoch["mask_start"]
                         mask_epochs_completed = current_epoch["mask_start"] + ((steps_trained / steps_to_train) * self.n_alt_train_amt)
                         step_lambda_lr(mask_optimizer,
                                        self.mask_lr_lambdas,
                                        mask_epochs_completed,
                                        self.mask_initial_lrs)

                # Check if we are done training; if so, exit the loop.
                if steps_trained >= steps_to_train:
                    break

            # Sample architectures.
            arch_fnames = self.sample_archs(current_epoch)
            saved_arch_fnames += arch_fnames

            # Write the saved loss values to a pickle file.
            save_loss_file = "loss_values_" + self.experiment_id
            with open(save_loss_file, "wb") as writefile:
                pickle.dump(loss_values, writefile)

        # Calculate time to complete search process.
        dnas_finish_time = time.time()
        with open(self.logfile_name, "a") as logfile_open:
            logfile_open.write(f"DNAS START TIME: {dnas_start_time}, DNAS FINISH TIME: {dnas_finish_time}, DNAS COMPLETION TIME: {dnas_finish_time - dnas_start_time}\n")
            logfile_open.flush()
