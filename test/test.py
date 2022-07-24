import os
import sys
import json
os.chdir('../src/')

from utils.datautils import *
from utils.modelutils import *

##Future editions will utilize pytest for practicality

print('building dataset dictionaries','\n')
##call class to get dictionaries
train_data = SequenceData(word2id, fam2label, config['data_config']['MAX_LEN'], config['data_config']['DATA_DIR'],"train")
train_dict = train_data.get_data_dictionaries()

def verify_labels_in_encoded_values(y: list, fam2label:dict):
    """This test verifies there are no extraneous values in the labels that do not exist in the label encoder"""
    assert set(list(Counter(y).keys())).issubset(list(fam2label.values())) == True, 'verify_labels_in_encoded_values FAILED - there are labels in y that dne in the encoder'
    print('verify_labels_in_encoded_values PASSED')

def verify_unique_labels(y: list, fam2labels: dict):
    """This verifies that the training data doesn't have more unique values than exists in the label encoder"""
    counter_object = Counter(y)
    keys = counter_object.keys()
    assert len(fam2labels) >= len(keys), 'failed, the number unique labels in training is greater than the expected number'
    print('verify_unique_labels PASSED: There are not more labels encoded in the data than are possible')    

def verify_feature_shapes(x: dict, max_len,word2id):
    """This test verifies that the feature sets are the expected size"""
    assert x['sequence'].shape[1:] == (max_len, len(word2id)), 'verified_feature_shapes failed - the shape of the features did not equal = (max_len,len(word2id))'
    print('Passed! Feature shape and configs for model Input layer are the same')    

def check_config_dtypes(config):
  count = 0
  broken_configs = []
  if isinstance(config['data_config']['BATCH_SIZE'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n data_config:BATCH_SIZE')

  if isinstance(config['data_config']['DATA_DIR'],str) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n data_config:DATA_DIR')

  if isinstance(config['data_config']['MAX_LEN'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n data_config:MAX_LEN')

  if isinstance(config['data_config']['TEST_BATCH_SIZE'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n data_config:TEST_BATCH_SIZE')

  if isinstance(config['model_config']['BATCH_SIZE'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:BATCH_SIZE')

  if isinstance(config['model_config']['DROPOUT'],float) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:DROPOUT')

  if isinstance(config['model_config']['D_RATE'],list) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:D_RATE')

  if isinstance(config['model_config']['EPOCHS'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:EPOCHS')

  if isinstance(config['model_config']['FILTERS'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:BATCH_SIZE')

  if isinstance(config['model_config']['L2_FACTOR'],float) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:L2_FACTOR')
    
  if isinstance(config['model_config']['LOSS'],str) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:LOSS')

  if isinstance(config['model_config']['MAX_LEN'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:MAX_LEN')

  if isinstance(config['model_config']['METRICS'],list) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:METRICS')

  if isinstance(config['model_config']['MODEL_DIR'],str) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:MODEL_DIR')

  if isinstance(config['model_config']['NUM_MODELS'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:NUM_MODELS')

  if isinstance(config['model_config']['NUM_RES_BLOCKS'],int) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:NUM_RES_BLOCKS')

  if isinstance(config['model_config']['OPT'],str) == True:
    pass
  else:
    count += 1
    broken_configs.append('\n model_config:OPT')
  broken_configs.append('\n')
  try:
    assert count == 0
    print('Config passes dtype test')
  except:
    print('config is broken, see following list')
    print(broken_configs)    
    
def main():

    with open('./config/config.json') as json_data_file:
      config = json.load(json_data_file)
    print(config) 
    
    train_data, train_targets = reader('train',config['data_config']['DATA_DIR'])
    fam2label = build_labels(train_targets)
    word2id = build_vocab(train_data)
    
    verify_labels_in_encoded_values(train_dict['target'], fam2label)
    
    verify_unique_labels(train_dict['target'], fam2label)
    
    verify_feature_shapes(train_dict, config['data_config']['MAX_LEN'],word2id)
    
    check_config_dtypes(config)
    
if __name__=="__main__":
    main()
    
    
