# protein_seq_classification

A repo for protein sequence classification based off [Using Deep Learning to Annotate the Protein Universe]("https://www.nature.com/articles/s41587-021-01179-w"), this project will use Deep Learning to build a classifier to labeled proteins based on their sequences to their family_ids. 

The data for this work came from [kaggle]("https://www.kaggle.com/datasets/googleai/pfam-seed-random-split"). 

This was largely done on google colab, as even with a TPU and High Ram run times it's very easy to time out on colab while dealing with this much data. For the training data there are almost 1.1 million sequences to classify into ~17930 labels. 

The Neural Networks (NN) used here use residual blocks with skip connections and bottle necks to learn the relationship between unaligned amiono acid sequences and their functional classifications across 17929 families. For our uses, the additional label denotes unknown protein structures. This arose due to a large class imbalance. 

Two possible models can be developed here via main.py using the config.json file. If the config lists 1 for the number of models, a ProtCNN model will be built. The paper reports this as having been the fastest NN to run on the sequence databases (in 2019). If the number of models is >1, main.py will build a ProtENN model, or an ensemble model made of ProtCNNs. As of now, we average the predictions made by the models in the ProtENN to predict the class. 

WARNING: Since ProtENN is meant to be an ensemble model, users should NOT assign an even number of models. 

## dir structure
├── Dockerfile
├── README.md
├── models
│   └── 2022_07_24_hr00_mm28_ss57
│       ├── simp_results.txt
│       └── verbose_results.txt
├── nb
│   ├── Sequence_and_fam_EDA.ipynb
│   ├── download_PFAM_to_drive.ipynb
│   ├── get_protCNN_inference.ipynb
│   └── get_protenn_model.ipynb
├── requirements.txt
└── src
    ├── config
    │   ├── config.json
    │   └── inf_config.json
    ├── inference.py
    ├── main.py
    ├── update_config.py
    ├── update_inference_config.py
    └── utils
        ├── datautils.py
        └── modelutils.py

## Usage

To use this git properly, one can either install Docker and download the docker image, place it inside some cloud resource and run via the image. This however will only run main.py, which reads in a base config file and generates either a ProtCNN or ProtENN. Users can also interact with this via a CLI (or notebook mimicing a CLI) calling main.py, inference.py, update_config.py, or update_inference_config.py. 

1. main.py trains, builds, and does inference as specified by the config.json file found in src/config/
2. inference.py does inference on pre-build models as specified by the inf_config.json found in src/config/
3. update_config.py updates the config.json for main.py and should be used for model development and hyperparameter tuning
4. update_inference_config.py updates the inf_config.json for inference.py and should only be used for doing model inference after a model has been build.

## nb
In the /nb/ directory that is on the same level of abstraction as /src/, there are several nbs detailing the evolution and development of the code found in /src/, as well as an Sequence_and_fam_EDA and a tensorboard_metrics notebook which detail the initial EDA and model tensorboard outputs respectively. 


##  Dataset
### Problem description
<b> THIS DESCRIPTION WAS TAKEN FROM [KAGGLE]('https://www.kaggle.com/datasets/googleai/pfam-seed-random-split') - ALSO NOTE: THIS DATA HAS A LISCENSE AND MAY NOT BE USED OUTSIDE OF PUBLIC FREE PROJECTS WITHOUT PERMISSION </b>.

Domains are functional sub-parts of proteins; much like images in ImageNet are pre segmented to contain exactly one object class, this data is presegmented to contain exactly and only one domain.

The purpose of the dataset is to repose the PFam seed dataset as a multiclass classification machine learning task.

The task is: given the amino acid sequence of the protein domain, predict which class it belongs to. There are about 1 million training examples, and 18,000 output classes.

### Data structure
This data is more completely described by the publication "Can Deep Learning Classify the Protein Universe", Bileschi et al.

### Data split and layout
The approach used to partition the data into training/dev/testing folds is a random split.

Training data should be used to train your models. Dev (development) data should be used in a close validation loop (maybe for hyperparameter tuning or model validation). Test data should be reserved for much less frequent evaluations - this helps avoid overfitting on your test data, as it should only be used infrequently.

### File content
Each fold (train, dev, test) has a number of files in it. Each of those files contains csv on each line, which has the following fields:

- sequence: HWLQMRDSMNTYNNMVNRCFATCIRSFQEKKVNAEEMDCTKRCVTKFVGYSQRVALRFAE
- family_accession: PF02953.15
- sequence_name: C5K6N5_PERM5/28-87
- aligned_sequence: ....HWLQMRDSMNTYNNMVNRCFATCI...........RS.F....QEKKVNAEE.....MDCT....KRCVTKFVGYSQRVALRFAE
- family_id: zf-Tim10_DDP

#### Description of fields:
- sequence: These are usually the input features to your model. Amino acid sequence for this domain.
There are 20 very common amino acids (frequency > 1,000,000), and 4 amino acids that are quite uncommon: X, U, B, O, Z.
- family_accession: These are usually the labels for your model. Accession number in form PFxxxxx.y (Pfam), where xxxxx is the family accession, and y is the version number. Some values of y are greater than ten, and so 'y' has two digits.
- family_id: One word name for family.
- sequencename: Sequence name, in the form " 𝑢𝑛𝑖𝑝𝑟𝑜𝑡𝑎𝑐𝑐𝑒𝑠𝑠𝑖𝑜𝑛𝑖𝑑/ startindex-$end_index". aligned_sequence: Contains a single sequence from the multiple sequence alignment (with the rest of the members of the family in seed, with gaps retained. <p> 
Generally, the family_accession field is the label, and the sequence (or aligned sequence) is the training feature.

This sequence corresponds to a domain, not a full protein.

