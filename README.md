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