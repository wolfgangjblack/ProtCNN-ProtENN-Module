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
# NUM_CLASSES: this is the number of unique labels we have in our training data
#
# INFERENCE_DIR: this is the directory where the model(s) live(s), path should be relative to /src/ directory
### 

data_config = {"DATA_DIR": './../../../PFAM_database/data/random_split/',
               "MAX_LEN": 120,
               "NUM_CLASSES": 17930,
               "INFERENCE_DIR": '../models/ensemble_model'}



config = {'data_config': data_config}

with open('./config/inf_config.json', 'w') as outfile:
    json.dump(config, outfile)