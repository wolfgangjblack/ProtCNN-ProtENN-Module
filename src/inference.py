import os
import sys
import json

from utils.datautils import *
from utils.modelutils import *

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
    inference_dir = config['data_config']['INFERENCE_DIR']
    num_classes = config['data_config']['NUM_CLASSES']
    
    test_dict = build_inference_data(config['data_config'])
    
    do_inference(test_dict, num_classes, inference_dir)
    
    print("Model Inference Done")
    
def main():
    """"The main function that will be run when the file is called. This will use the functions in the utils and those defined above to process the data, build the model(s), and output the results using the fields within the config"""
    
    print('warning: this is expected the inf_config.json with a pre-specified inference_dir')
    
    with open('./config/inf_config.json') as json_data_file:
        config = json.load(json_data_file)
    
    print(config)
    
    get_model_inference(config)
    
if __name__=="__main__":
    main()
