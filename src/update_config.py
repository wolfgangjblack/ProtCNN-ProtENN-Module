## Author: Wolfgang Black
# Date_modified: 7/23/22
# ------------------------
#This notebook is meant to overwrite the config in 'src/config/'. 
#
#If a user wants to change parameters for the data or model developement - here is where they can do it. 
# -------------------------
#
# This file is meant to be read in src and has been written as though it were in my colab
#
import os
import json

# -------------------------
#data_config entries
# DATA_DIR: this is the directory where the data lives, path should be relative to /src/ directory
#
# MAX_LEN: this is the max allowable length for sequences. sequences range in length, with 100 to 120 being common choices - 
#     this code pads any sequences LT MAX_LENGTH at the end of the sequence
#     THIS CAN BE CONSIDERED TO BE A TRAINING PARAMETER
#
# TEST_BATCH_SIZE: this is the batch size that will be used in the batch command when the test feature is turned 
#     into batch tensor objects - this is meant to be lower for memory purposes
#
# TEST_RANGE: this is a list that will be used during inference when testing scripts due to limited memory on systems. 
#      This grabs the data for inference between TEST_RANGE0] and TEST_RANGE[1]
### 

data_config = {"DATA_DIR": './../../../PFAM_database/data/random_split/',
               "MAX_LEN": 120,
               "TEST_BATCH_SIZE": 100,
               "TEST_RANGE": [0,10000]}

# -------------------------
#model_config entries
# NUM_MODELS: this is the number of models to build during model development.
#     1 - this will create a single ProtCNN
#     >1 - this will create an ensemble ProtENN model NOTE: DO NOT DO EVENS (ADD TEST HERE)
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
# MODEL_DIR: this is the parent model directory, meant to be on the same level of abstraction as /src/ and relative to /src/
#
# NUM_RES_BlOCKS: this is an int and will give the number of res blocks per ProtCNN, 2 or 3 are a common choice
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
# FILTERS: this is an int and will specify the number of filters in EACH convolutional layer in the model
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
# D_RATE: this is a list of dilution rates where each int in the list belongs to its own res block. 2 and 3 are common choices
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
# MAX_LEN": inherited from data_config, this is the max allowable length for the sequence
#     THIS CAN BE CONSIDERED TO BE A TRAINING PARAMETER
#
# DROPOUT: a single float (0<x<1) meant to denote the amount of drop out before the end of a single ProtCNN block. 
#      For now, this will be a single value for each ProtCNN model build.
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
# L2_FACTOR: the kernel_regularization argument for L2 regularization, this is a float - typically 0.0001
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
# OPT: this is short for optimizer, and will be used in model.compile. Tested with 'adam' but no other optimizer
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
# LOSS: this is the loss function used in model.compile. 
#      Since we have a multiclass classifier use 'sparse_categorical_crossentropy',
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
#"METRICS": these are the metrics which will be reported during model training, tested only with ['accuracy']
#
#"EPOCHS": this is the number of times each model will run through the training/validationd data.
#    For cloud computing set to gte 10
#    Note: on google colab with TPU 1 epoch takes roughly 1 hour
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
# BATCH_SIZE: this is the batch size that will be used in the batch command when the training, validation datasets are turned 
#     into batch tensor objects
#     THIS CAN BE CONSIDERED TO BE A MODEL HYPERPARAMETER
#
### NOTE: Looking at /src/utils/modelutils.py the build_train_model function also has vocab_size and num_classes items in the config file. These are added by the build_training_datasets function (found in /src/utils/datautils.py)

model_config = {"NUM_MODELS": 1,
                "MODEL_DIR": '../models/',
                "NUM_RES_BLOCKS": 3,
                "FILTERS": 128,
                "D_RATE": [2, 3, 3],
                "MAX_LEN": data_config['MAX_LEN'],
                "DROPOUT": 0.5,
                "L2_FACTOR": 0.0001,
                "OPT": 'adam', 
                "LOSS": 'sparse_categorical_crossentropy',
                "METRICS": ['accuracy'],
                "EPOCHS": 5,
                "BATCH_SIZE": 256}

config = {'data_config': data_config,
          'model_config': model_config}

with open('./config/config.json', 'w') as outfile:
    json.dump(config, outfile)
