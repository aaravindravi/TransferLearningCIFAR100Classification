"""
Aravind Ravi
Transfer Learning
Classification - ResNet 50 Features Transferred - CIFAR 100
Submission files have been trained with 100% of training data after performing validation on 80-20 split of data
NOTE: Run the feature extraction file before running this script for classification
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import applications
from keras.preprocessing import image
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Input,Dense,Flatten,Dropout,Activation,BatchNormalization
from keras.models import Sequential
from keras import regularizers


#Loading the Features
X_train = np.load("train_dataavg_poolresnet_data.npy")
X_test = np.load("test_dataavg_poolresnet_data.npy")

#Loading the labels of training data
with open('train_data/train_data', 'rb') as f:
    train_data = pickle.load(f)
    train_label= pickle.load(f)

#Convert the labels to one-hot encoding
trainImageLabels = keras.utils.to_categorical(train_label, num_classes=100)

#Train Test Split 80%-20%
x_tr,x_ts,y_tr,y_ts = train_test_split(X_train, trainImageLabels, test_size=0.2,random_state=1)

#Creating a Deep Model for classifying the features of CIFAR-100 extracted from ResNet-50 trained on ILSVRC Dataset
model = Sequential()
model.add(Dense(2048, input_dim=2048, kernel_initializer="uniform")) #Fully connected layer (as output feature size of ResNet is 2048
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))  
model.add(Dense(512, kernel_initializer="uniform"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))  
model.add(Dense(100))
model.add(Activation("softmax"))
model.summary()

#Optimizer - Stochastic Gradient Descent
sgd = optimizers.SGD(lr=0.01, decay=0.00001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_tr,y_tr,batch_size=128,epochs=30,validation_data=(x_ts, y_ts),verbose=1)

#Validation Accuracy
score = model.evaluate(x_ts, y_ts,verbose=1)
print(score)

#Save the trained Model
model.save('trainedModel.h5')

#Classify the test data (Submission files have been trained with 100% of training data after validating on 80-20)
predictions_ts = (model.predict(X_test))
class_result=np.argmax(predictions_ts,axis=-1)

np.savetxt("submission_labels.csv", class_result, delimiter=",")
