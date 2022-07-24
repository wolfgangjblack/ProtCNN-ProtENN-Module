import os
import sys
import json

from utils.datautils import *
from utils.modelutils import *

def get_training_sets(config: dict):
    """This function is used by main to call and define the training and validation datasets
    Args:
        config: dict
            - this funciton uses the config['data_config'] sub dictionary and calls the datautils function build_training_datasets
    Outputs:
        train_ds: tf Batch Tensor
            - this is the training features and labels as a Batch Tensor 
        validation_ds: tf Batch Tensor
            - this is the validation features and labels as a Batch Tensor
        config: dict
            - this is the same full config initially read in to the function, however the sub dictionary 'model_config' has been given two new items
                - NUM_CLASSES: int
                    - the number of labels as determined by the train_ds
                - VOCAB_SIZE: int
                    - the max vocab size used in the features, this can be thought of as the max number of ohe indicies
    """
    train_ds, validation_ds, config['model_config']['NUM_CLASSES'], config['model_config']['VOCAB_SIZE'] = build_training_datasets(config['data_config'])
    return train_ds, validation_ds, config

def build_models(config:dict , train_ds, validation_ds) -> dict:
    """This function is used by main to build the model(s) as described by the config
    Arg:
        config: dict
            - this funciton uses the config['model_config'] sub dictionary and calls the modelutils function build_train_model
        train_ds: tf Batch Tensor
            - this is the training features and labels as a Batch Tensor 
        validation_ds: tf Batch Tensor
            - this is the validation features and labels as a Batch Tensor
    Output:
        config: dict
            - this is the same full config initially read in to the function, however the sub dictionary 'data_config' has been given a new item
                - inference_dir is the directory where the models have been saved by the modelutils function build_train_model. This is also the directory where results from inferences will be saved
    """
    config['data_config']['INFERENCE_DIR'] = build_train_model(config['model_config'], train_ds, validation_ds)
    return config

def get_model_inference(config):
    """This function is used by main to perform inference on the models trained previously during main and save the results to a results directory
    Arg:
        config: dict
            - this funciton uses the config['data_config'] sub dictionary and calls the datautils function build_inference_data and the modelutils function do_inference
    Output: NOTE - THIS FUNCTION DOES NOT RETURN ANYTHING, IT SIMPLY SAVES THE RESULTS
        results_dir: directory
            - this model generates a directory inside the inference_dir and stores the 
        verbose_results: txt
            -classification_report in its entirety
        simplified_results: txt
            - an abbreviated version of classification_report which only shows the accuracy and micro/macro precision, recall, and f1 scores
    """
    test_dict = build_inference_data(config['data_config'])
    
    do_inference(test_dict, config)
    
    print("Model Inference Done")
    
def main():
    """"The main function that will be run when the file is called. This will use the functions in the utils and those defined above to process the data, build the model(s), and output the results using the fields within the config"""
    with open('./config/config.json') as json_data_file:
        config = json.load(json_data_file)
    
    print(config)
    
    train_ds, validation_ds, config= get_training_sets(config)
    
    config = build_models(config, train_ds, validation_ds)
    
    get_model_inference(config)
    
    with open(config['data_config']['INFERENCE_DIR']+'final_config.json', 'w') as outfile:
        json.dump(config, outfile)
    
if __name__=="__main__":
    main()
