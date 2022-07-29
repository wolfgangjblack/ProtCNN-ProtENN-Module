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
# MAX_LEN: this is the max allowable length for sequences. sequences range in length, with 100 to 120 being common choices - this
#     code pads any sequences LT MAX_LENGTH at the end of the sequence
#
# TEST_BATCH_SIZE: this is the batch size that will be used in the batch command when the test feature is turned 
#     into batch tensor objects - this is meant to be lower for memory purposes
#
# INFERENCE_DIR: this is the directory where the model(s) live(s), path should be relative to /src/ directory
#
# TEST_RANGE: this is a list that will be used during inference when testing scripts due to limited memory on systems. 
#      This grabs the data for inference between TEST_RANGE0] and TEST_RANGE[1]
### 

data_config = {"DATA_DIR": './../../../PFAM_database/data/random_split/',
               "MAX_LEN": 120,
               "TEST_BATCH_SIZE": 100,
               "INFERENCE_DIR": '../models/ensemble/',
               "TEST_RANGE": [0,10000]}



config = {'data_config': data_config}

with open('./config/inf_config.json', 'w') as outfile:
    json.dump(config, outfile)
