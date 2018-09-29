# HAB Deep Learning Classifications

This code is for generating classfication scores for HAB databases

There are two basic classification methods:

1. Extract features from each frame with a ConvNet, passing the sequence to an RNN, in a separate network
2. Extract features from each frame with a ConvNet and pass the sequence to an MLP

## Requirements

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

## Getting the data

The data is extracted using a MATLAB script and deposited into the CNNIms
directory (one jpg per time stamp).

## Extracting features

For the two models (`lstm` and `mlp`) features are firstly extracted from each jpg image using the 
`extract_features.py` script. 

## TODO

- [ ] Integrate other CNN models (e.g. VGG)
- [ ] Create "whole model" with one CNN for each modality (fine tuned)
- [ ] Try removing more layers of the CNNs

