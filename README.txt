**************************************************
Aravind Ravi

**************************************************
README
**************************************************

Methodology: Transfer Learning
Requirement: GPU for faster feature extraction and training
Dataset: CIFAR-100 (Download from the https://www.cs.toronto.edu/~kriz/cifar.html)

Folder contains two files:
1. featureExtraction_resNet.py
2. resnetTransferLearning_Classification.py

Tasks of feature extraction and classification have been separated due to computational constraints

Note: 
1. Run the first file before running the second file
2. Keep the train dataset in the current path inside another folder train_data (as the code looks for train_data/train_data)
3. Keep the test dataset in the current path inside another folder test_data (as the code looks for test_data/test_data)

Results:
The model was validated on a split of 80%-20% Train-Test with approximately 76% accuracy

Model was trained on 100% of the data for the prediction task - 77.4% (Can be validated with labels from original website)

**************************************************
featureExtraction_resNet.py
**************************************************
This file is used to extract features of the CIFAR-100 dataset.
For feature extraction, the Resnet-50 Model is used. The model was trained on the "ILSVRC Dataset" and 
is used here for extracting deep features.

The output for each image is a feature vector of length 2048. Features are extracted from the global average pooling layer 
These features are used further to train a Deep Neural network for classification.


********************************************************
resnetTransferLearning_Classification.py
********************************************************
This file contains a Deep Neural Network to classify the features extracted from the ResNet-50 for the CIFAR-100 dataset
The input layer takes the feature vector of length 2048. 
The last classification layer is 100-way classification layer for the CIFAR-100 dataset.

The classifier was trained on 80% of the training data and validated on 20% of the remaining data.

For the submission of the final labels, the classifier was trained on 100% of the training data and 
tested on the 10000 test data images provided in the data challenge.


