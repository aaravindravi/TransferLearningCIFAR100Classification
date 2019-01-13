"""
Aravind Ravi
Transfer Learning
Feature Extraction - Res Net 50 - CIFAR 100
ResNet50 pre-trained on ILSVRC dataset - Features from Global Average Pooling Layer
"""
import pickle
import numpy as np
import cv2
from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

#Loading the Dataset
with open('train_data/train_data', 'rb') as f:
    train_data = pickle.load(f)
    train_label= pickle.load(f)
with open('test_data/test_data', 'rb') as f:
    test_data = pickle.load(f)

#Residual Network with ILSVRC weights
base_model = applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#Debug
base_model.summary()
#Extract from the average pooling layer
layers_to_extract = ["avg_pool"]

#Select the features from average pooling layer
model = Model(input=base_model.input, output=base_model.get_layer(layers_to_extract[0]).output)


#To extract the features from the selected layer of ResNet50 Net
layer_num=0
feats=[]
for img_count in range (0,50000): #Change to 10000 for test data
	print(img_count)
	
	#Pre-processing
	image1 = np.zeros((32,32,3),dtype=np.uint8)
	image1[...,0] = np.reshape(train_data[img_count,:1024],(32,32)) #replace with test_data for test data features
	image1[...,1] = np.reshape(train_data[img_count,1024:2048],(32,32)) #replace with test_data for test data features
	image1[...,2] = np.reshape(train_data[img_count,2048:3072],(32,32)) #replace with test_data for test data features
	image1 = cv2.resize(image1,(224,224))
	x_in = image.img_to_array(image1)
	x_in = np.expand_dims(x_in, axis=0)
	x_in = preprocess_input(x_in)
	
	#Feature Extraction
	features = model.predict(x_in)
	features = features.flatten()
	feats.append(features)
	features_arr = np.char.mod('%f', features)
	
feature_list = np.squeeze(np.asarray(feats))
np.save("train_data"+layers_to_extract[layer_num]+"resnet_data.npy",feature_list)

# Test Dataset

layer_num=0
feats=[]
for img_count in range (0,10000): #Change to 10000 for test data
	print(img_count)
	
	#Pre-processing
	image1 = np.zeros((32,32,3),dtype=np.uint8)
	image1[...,0] = np.reshape(test_data[img_count,:1024],(32,32)) #replaced with test_data for test data features
	image1[...,1] = np.reshape(test_data[img_count,1024:2048],(32,32)) #replaced with test_data for test data features
	image1[...,2] = np.reshape(test_data[img_count,2048:3072],(32,32)) #replaced with test_data for test data features
	image1 = cv2.resize(image1,(224,224))
	x_in = image.img_to_array(image1)
	x_in = np.expand_dims(x_in, axis=0)
	x_in = preprocess_input(x_in)
	
	#Feature Extraction
	features = model.predict(x_in)
	features = features.flatten()
	feats.append(features)
	features_arr = np.char.mod('%f', features)
	
feature_list = np.squeeze(np.asarray(feats))
np.save("test_data"+layers_to_extract[layer_num]+"resnet_data.npy",feature_list)


