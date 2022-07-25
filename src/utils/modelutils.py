import os

import tensorflow as tf
import keras
import numpy as np
from datetime import datetime

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Add, MaxPooling1D, Flatten, Dropout, Dense, Activation, BatchNormalization, Conv1D
from tensorflow.keras import Model, Input, Sequential
from keras.regularizers import l2

from sklearn.metrics import classification_report

def get_date_st():
    """This function returns the current datetime with hour, minutes, and seconds for record keeping
    """
    now = datetime.now()
    return now.strftime('%Y_%m_%d_hr%H_mm%M_ss%S/')

def residual_block(prev_layer, filters: int, d_rate: list):
    """The residual block contains the batch normalization and convolutional layers used in the ProtCNN model.
    This includes a skip connection with ADD() at the bottom allowing the input to this model to be added to the end
  Args:
      prev_layer: tf.keras.layer
          - this is the input layer to the res block, note: this may be the output of another res block or some other layer
      FILTERS: int
          - the number of filters in both convolutional layers
      D_RATE: int
          - the dilution rate used in the first conv layer of the res block
  """
    bn1 = BatchNormalization()(prev_layer)
    act1 = Activation('relu',)(bn1)
    conv1 = Conv1D(filters, 1, dilation_rate = d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)
    
    #bottleneck convolution
    bn2 = BatchNormalization()(conv1)
    act2 = Activation('relu')(bn2)
    conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(act2)

    #skip connection
    x = Add()([conv2, prev_layer])

    return x

def get_protCNN_model(num_res_blocks: int,
                      max_len: int, vocab_size: int, num_classes: int,
                      filters: int, d_rate: list, dropout: float, l2_factor: float,
                     opt: str, loss: str, metrics: list):
    """This function builds the ProtCNN model. It can have multiple residual blocks, 
    but expects all conv layers to have the same filters and the output layer to use l2_regularization
    Args:
      num_res_blocks: int
        - the number of residual blocks in the model
      max_len: int
        - this is the max sequence length and is a training hyperparameter. We'll leave to 120 for the time being
      vocab_size: int
        - this is the max number of possible ints in the vocab, passed from the word2id
      num_classes: int
        - this is the max_number of possible ints for the labels, passed from fam2label
      filters: int
        - this is the number of filters in each convolutional layer
      d_rate: list of ints
        - this is a list of dilution rates. It must be of size NUM_RES_BLOCKS
      dropout: int
        - this is the rate of dropout the model experiences just before the output layer
      l2_factor: float 
        - the l2_factor in the regularization kernal for the output layer
      Opt: str
        - this is the string representing which optimizer to use for the model. Tested with 'adam'
      loss: str
        - this is the loss function which the model optimizes, since we've got 17930 labels we'll use
         'sparse_categorical_crossentropy'
      metrics: list
        - these are the metrics the model reports as it trains.
    Outputs:
        model: tf.model()
          - this is the compiled ProtCNN architecture 
    """
    # input
    x_input = Input(shape=(max_len, vocab_size))

    #initial conv
    x = Conv1D(filters, 1, padding='same')(x_input) 

    # per-residue representation
    for i in range(num_res_blocks):
        x = residual_block(x, filters, d_rate[i])
    
    x = MaxPooling1D(3)(x)
    x2 = Dropout(dropout)(x)

    # softmax classifier
    x3 = Flatten()(x2)
    x_output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_factor))(x3) 

    model = Model(inputs=x_input, outputs=x_output)

    model.summary()
    
    model.compile(optimizer= opt, loss = loss, metrics = metrics)
    return model

def build_train_model(config: dict, 
                      train_ds, val_ds) -> str:
    """This function builds the ProtCNN model(s) specified by the config, trains the model, and
    saves the results while producing a config entry meant to be read for inference. Note, this uses for loops over num_models to construct, train, and save ProtCNN models
    Args:
        config: dict
            - this is the config['model_config'] sub_dictionary which contains all the variables necessary for modeling
              num_models: int
                  - this is the number of ProtCNN models to build, meant to be between >=1
              model_dir: str
                  - This is the directory relative to /src/ where the model data should be saved
             num_res_blocks: int
                 - the number of residual blocks in the model
             max_len: int
                 - this is the max sequence length and is a training hyperparameter. We'll leave to 120 for the time being
             vocab_size: int
                 - this is the max number of possible ints in the vocab, passed from the word2id
             num_classes: int
                 - this is the max_number of possible ints for the labels, passed from fam2label
             filters: int
                 - this is the number of filters in each convolutional layer
             d_rate: list of ints
                 - this is a list of dilution rates. It must be of size NUM_RES_BLOCKS
             dropout: int
                 - this is the rate of dropout the model experiences just before the output layer
             l2_factor: float 
                 - the l2_factor in the regularization kernal for the output layer
             Opt: str
               - this is the string representing which optimizer to use for the model. Tested with 'adam'
             loss: str
               - this is the loss function which the model optimizes, since we've got 17930 labels we'll use
                 'sparse_categorical_crossentropy'
             metrics: list
               - these are the metrics the model reports as it trains.
    Output:
        inference_dir: str
            - this outputs a string directory relative to the /src where the models have been saved. This will be used for inference
                      """
    num_models = config['NUM_MODELS']
    model_dir = config['MODEL_DIR']
    num_res_blocks = config['NUM_RES_BLOCKS']
    filters = config['FILTERS']
    d_rate = config['D_RATE']
    max_len = config['MAX_LEN']
    dropout = config['DROPOUT']
    l2_factor = config['L2_FACTOR']
    opt = config['OPT']
    loss = config['LOSS']
    batch_size = config['BATCH_SIZE']
    epochs = config['EPOCHS']
    metrics = config['METRICS']
    vocab_size = config['VOCAB_SIZE']
    num_classes = config['NUM_CLASSES']
    if num_models > 1:
        model_dir = model_dir+'ensemble/'+get_date_st()
    else:
        model_dir = model_dir+get_date_st()
  
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for i in range(num_models):
        ##Trains num_models ProtCNN models
        print(f'training model {i}')
        #Constructs model using the get_protCNN_model
        model = get_protCNN_model(num_res_blocks, max_len, vocab_size, num_classes, filters , d_rate, dropout, l2_factor, opt, loss, metrics)
        ##We've reshuffled here in case of ensemble modeling.
        history = model.fit(train_ds.shuffle(True).batch(batch_size), 
                            epochs = epochs,
                            batch_size = batch_size,
                            validation_data = val_ds.batch(batch_size))
                            
        if num_models > 1:
            model.save(model_dir+'ProtENN_model_'+str(i)+'{}_epoch_model.h5'.format(epochs))
        else:
            model.save(model_dir+'ProtCNN_model_'+'{}_epoch_model.h5'.format(epochs))
    print('\nFinished Saving Model(s)\n')
    
    return model_dir

def do_inference(test_dict: dict, config: dict):
    """This function does inference on whatever model(s) are in the inference_dir, which is the output of build_train_model.
    Args:
        test_dict: dict
            - this is the dictionary containing test data sequences and targets, produced by build_inference_data
        batch_size: int
            - this is the batch_size used in inference when storing the test_dataset in memory
        inference_dir: str
            - this is the directory where the model(s) live. This can be either output during the model training or specified in inf_config if the user is just doing inference on (a) model(s)
    Outputs:
        results_dir: directory
            - this model generates a directory inside the inference_dir and stores the 
        verbose_results: txt
            -classification_report in its entirety
        simplified_results: txt
            - an abbreviated version of classification_report which only shows the accuracy and micro/macro precision, recall, and f1 scores
    """
    ##initialize
    batch_size = config['data_config']['TEST_BATCH_SIZE']
    inference_dir = config['data_config']['INFERENCE_DIR']
    num_classes = config['modle_config']['NUM_CLASSES']
    test_range = config['data_config']['TEST_RANGE']
    if len(test_range) < 2:
        pass
    elif len(test_range) == 2:
	test_dict = {'sequence': test_dict['sequence'][test_range[0]:test_range[1]], 'target': test_dict['target'][test_range[0]:test_range[1]]}

    test_dataset = tf.data.Dataset.from_tensor_slices(test_dict['sequence']).batch(batch_size)
    y_preds = np.zeros((len(test_dir['target']), num_classes))
    y_true = test_dict['target']

    k = 0 #counter for averaging
    for i in [i for i in os.listdir(inference_dir) if '_model' in i]:
        print(i)
        ##Load in model and add predictions
        model = tf.keras.models.load_model(inference_dir+i)
        y_preds += model.predict(test_dataset)
        k += 1
        print("finished loading predictions")
   
    ##average y_preds incase of ensemble method, if single ProtCNN, this divides by 1
    y_preds = y_preds/k
    
    ##get the index of the max
    y_preds = np.argmax(y_preds, axis = 1)
    
    class_rep = classification_report(y_true, y_preds)
    
    print(class_rep)
    
    results_dir = inference_dir+'results/'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    ##Write full classification report to a txt for review later
    with open(results_dir+"verbose_results.txt", "w") as external_file:
        print(class_rep, file=external_file)
        external_file.close()
        
    print('\n\n')

    abrev_class_rep = class_rep.split('\n\n')[0]+'\n\n'+class_rep.split('\n\n')[-1]
    
    print(abrev_class_rep)
    ##Write abbreviated class report to a txt for review later
    with open(results_dir+"simp_results.txt", "w") as external_file:
        print(abrev_class_rep, file=external_file)
        external_file.close()
