import os
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

def get_date_st()-> str:
    """This function returns the current datetime with hour, minutes, and seconds for record keeping
    Output:
        now.strftime: str
            - this is a string with the format %Y_%m_%d_hr%H_mm%M_ss%S/. It's followed by a '/' as we'll use it as the suffix of  a folder
    """
    now = datetime.now()
    return now.strftime('%Y_%m_%d_hr%H_mm%M_ss%S/')


def reader(partition: str, data_path: str) -> pd.Series:
    """This function reads in a list of csv files found in data_path/partition and returns two pd.Series
  Args:
    partition: str
      - this is designed to accept strs 'train', 'test', or 'dev'
    data_path: str
      - this is designed to be the directory to the data downloaded from kaggle and expects to see the subdirectory options detailed in partition
  Output:
    all_data['sequence']: pd.Series (strings)
      - this pd.Series is a column of sequence data from the csvs. This is typically the main feature in protein classification Amino acid sequence 
          for this domain. There are 20 very common amino acids (frequency > 1,000,000), and 4 amino acids that are quite uncommon: X, U, B, O, Z.
    all_data['family_accession]: pd.Series
    - this pd.Series is a column of family data from the csvs. Accession number in form PFxxxxx.y (Pfam), where xxxxx is the family accession, and y is the version
     number. Some values of y are greater than ten, and so 'y' has two digits.
  """ 
    data = []

    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
          data.append(pd.read_csv(file, index_col=None, usecols=["sequence","family_accession"]))
    all_data = pd.concat(data)

    return all_data["sequence"], all_data["family_accession"]


def build_labels(targets: pd.Series) -> dict:
    """build_labels returns fam2label, a dictionary containing keys of unique family_accession and values of frequency
      Args:
        targets: pd.Series (str)
          - targets is a pd.Series produced by reader of the datas family_accession
      Output:
        fam2label: dict
          - this dictionary has keys of unique family_accession and values of the frequency that each label appears
      """
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0
    print(f"There are {len(fam2label)} labels.",'\n')
    return fam2label


def get_amino_acid_frequencies(data: pd.Series) -> pd.DataFrame():
    """get_amino_acid_frequencies uses a counter to count recurring sequences within the data
  Args:
    data: pd.Series
      - This data is the sequence data made into a pd.Series via the previous reader() function
  Output:
    pd.DataFrame()
      - this dataframe contains the keys and values of the counted sequence data
  """
    aa_counter = Counter()
    for sequence in data:
        aa_counter.update(sequence)
    return pd.DataFrame({'AA': list(aa_counter.keys()),
                       'Frequency': list(aa_counter.values())})
  
def build_vocab(data: pd.Series) -> dict:
    """This function builds the vocabulary to be used in mapping the sequences to ints.
    Args:
        data: pd.Series
            - This data is the sequence data
    Output:
        word2id: dict
            - this dictionary contains the codes to take a letter to an int
    """
    voc = set()
    rare_AAs = {'X', 'U', 'B', 'O', 'Z'}
    for sequence in data:
        voc.update(sequence)

    unique_AAs = sorted(voc - rare_AAs)
    #build mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=1)}
    word2id['<unk>'] = 0
    print(f"AA dictionary formed. the length of dictionary is: {len(word2id)}.",'\n')
    return word2id


class SequenceData():
    """This class takes the sequence data and transforms it into a dictionary of ohe array and transforms the targets to their int labels. Next version we'll improve this to self initialize the dictionaries as opposed to the need to call the method. 
    """
    def __init__(self, word2id, fam2label, max_len, data_path, split):
        """"Here we inherit the necessary dictionaries and values to change the data from str to ints"""
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len
        self.split = split
        self.data_path = data_path
        self.data, self.labels = reader(split, data_path)

    def preprocess(self):
        """this function is what transforms the data from reader into the one_hot_encoded arrays for the sequence features and
        into the int for the targets.
        Args:
            Self:
                - this inherits the data and labels created by reader, as well as the dictionaries for
                word2id and fam2label
        Outputs:
            data_encode_ohe: dict
                - this is the sequence data encoded into ints based off the word2id
            targets: list
                - this is the family_ids put into the int form based off the fam2label
        """
        data = list(self.data.values)
        labels = list(self.labels.values)

        data_encode = []
        targets = []
        for i in range(len(data)):
            data_encode.append([self.word2id.get(i,self.word2id['<unk>']) for i in data[i]])
            targets.append(self.fam2label[labels[i]])

        data_encode = pad_sequences(data_encode, maxlen = self.max_len, padding = 'post', truncating = 'post')  
        data_encode_ohe = to_categorical(data_encode)

        return data_encode_ohe, targets
    
    def get_data_dictionaries(self) -> dict:
        """The final method and what should probably be fed into the __getitem__ method later, this uses the preprocess
        function to generate the sequences and target for the dataset
        Outputs:
            dict: dict
                ['sequence']: this is the int one hot encoded sequence generated by preprocess for the feature data
                ['targets']: this is the int label generate by preprocess for the target data
        """
        sequences, targets = self.preprocess()

        return {'sequence': sequences, 'target': targets}

def build_training_datasets(config: dict):
    """This function generates the train and validation datasets necessary for model training.
    If doing inference, use build_inference_data
    Args:
        config:dict
            - this config has the follow arguments as key,value pairs. Note: this is a sub_dictionary created by 
                /src/update_config.py
          data_dir: str directory
              - this is the directory where the data lives, if on google colab will need to open drive
          max_len: int
              - this is the max sequence length and is a training hyperparameter. We'll leave to 120 for the time being
          batch_size: int
              - this is the batch_size for the tensorflow and how the data is stored in memory
    Outputs:
        train_dataset: tensorflow BatchTensor
          - the training data put into a tensorflow batch dataset. 
            Note: this is shuffled
        validation_dataset: tensorflow BatchTensor
          - the validation data put into a tensorflow batch dataset
      """
    ##Get dicts for sequence/family mapping based off training data
    
    data_dir = config['DATA_DIR']
    max_len = config['MAX_LEN']    
    print("building artifacts for training",'\n')
    
    train_data, train_targets = reader('train',data_dir)
    fam2label = build_labels(train_targets)
    word2id = build_vocab(train_data)
    
    print('building dataset dictionaries','\n')
    ##call class to get dictionaries
    train_data = SequenceData(word2id, fam2label, max_len, data_dir,"train")
    train_dict = train_data.get_data_dictionaries()
    
    dev = SequenceData(word2id, fam2label, max_len, data_dir,"dev")
    dev_dict = dev.get_data_dictionaries()
    
    ##Use Tensorflow to put into tensordataset
    print("building tensorflow datasets",'\n')
    train_dataset = tf.data.Dataset.from_tensor_slices(
      (train_dict['sequence'], train_dict['target']))
    
    validation_dataset = tf.data.Dataset.from_tensor_slices(
      (dev_dict['sequence'], dev_dict['target']))
          
    print("finished building training sets",'\n')
    return train_dataset, validation_dataset, len(fam2label), len(word2id)

def build_inference_data(config: dict) -> dict:
    """This function generates the test dataset necessary for model inference.
    If doing model training/dev, use build_training_datasets
    Args:
        config: dict
            - this is expecting config['data_config'] and contains the necessary following peices
          data_dir: str directory
              - this is the directory where the data lives, if on google colab will need to open drive
          max_len: int
              - this is the max sequence length and is a training hyperparameter. We'll leave to 120 for the time being
    Outputs:
        test_dict: dict
          - this is the dictionary of sequences and family_id ints output from the Sequence class. 
  ----------------------------------------------------
  Note: Future improvements to the general code can save off fam2label and word2id as artifacts which are recalled
  during the dataset developement. This will streamline both build_data functions considerably
  """
    data_dir = config['DATA_DIR']
    max_len = config['MAX_LEN']
    
    ##Get dicts for sequence/family mapping based off training data
    print("\ngetting training artifacts")
    train_data, train_targets = reader('train',data_dir)
    fam2label = build_labels(train_targets)
    word2id = build_vocab(train_data)
    
    ##call class to get dictionaries
    test = SequenceData(word2id, fam2label, max_len, data_dir, "test")
    test_dict = test.get_data_dictionaries()
    print("finished building data for inference\n")

    return test_dict
    