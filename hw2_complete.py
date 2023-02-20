### Add lines to import modules as needed

import tensorflow as tf 
import keras 
import numpy as np 
from PIL import Image
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

## 
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def build_model1():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(32, 32, 3)),
      
      keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      
      keras.layers.Conv2D(64, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      
      keras.layers.Conv2D(128, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      
      keras.layers.Conv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      
      keras.layers.Conv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      
      keras.layers.Conv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      
      keras.layers.Conv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      
      keras.layers.MaxPooling2D(pool_size = (4, 4), strides = (4, 4)),
      
      keras.layers.Flatten(),
      
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(10)
      
      
  ])
# Add code to define model 1.
  return model

def build_model2():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(32, 32, 3)),
      
      keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
     # keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      
      keras.layers.SeparableConv2D(64, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      #keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      
      keras.layers.SeparableConv2D(128, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      #keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      
      keras.layers.SeparableConv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      #keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      
      keras.layers.SeparableConv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
     # keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      
      keras.layers.SeparableConv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
     # keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      
      keras.layers.SeparableConv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
     # keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      
      keras.layers.MaxPooling2D(pool_size = (4, 4), strides = (4, 4)),
      
      keras.layers.Flatten(),
      
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(10)
      
      
  ])

# Add code to define model 1.
  return model

def build_model3():
  # model = tf.keras.Sequential([
    inputshape =  keras.Input(shape=(32,32,3))
   # inputs = tf.keras.Input(shape=inputshape)
    
    y = keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = "same")(inputshape)
    x = keras.layers.BatchNormalization()(y)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.25)(x) 
    
    #y = keras.layers.DepthwiseConv2D((1,1),activation = "relu", padding = "same")(y)
    y = keras.layers.add((x,y))
    
    
    x = keras.layers.Conv2D(64, (3,3), strides = (2,2), padding = "same")(y)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    
    x = keras.layers.Conv2D(128, (3,3), strides = (2,2), padding = "same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.25)(x)
    
    #skip layer here 
    y = keras.layers.Conv2D(128,(1,1), strides = (4,4), activation = "relu", padding = "same")(y)
    y = keras.layers.add((x,y))

    
    x = keras.layers.Conv2D(128, (3,3), padding = "same")(y)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.25)(x)
       
    x = keras.layers.Conv2D(128, (3,3), padding = "same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.25)(x)
       
    #skip layer here 
    y = keras.layers.add((x,y))
    
    
    x = keras.layers.Conv2D(128, (3,3), padding = "same")(y)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.25)(x)
       
    x = keras.layers.Conv2D(128, (3,3), padding = "same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.25)(x)
      
    x = keras.layers.MaxPooling2D(pool_size = (4, 4), strides = (4, 4))(x)
      
    x = keras.layers.Flatten()(x)
      
    x = keras.layers.Dense(128, activation="relu")(x)
    out =  keras.layers.Dense(10)(x)
      
      
 # ])
    model = keras.Model(inputs = inputshape, outputs = out)
  # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
    return model

def build_model50k():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(32, 32, 3)),
      
      keras.layers.Conv2D(32, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      keras.layers.Dropout(0.25),
      
      keras.layers.SeparableConv2D(32, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      keras.layers.Dropout(0.25),
      
      keras.layers.SeparableConv2D(64, (3,3), strides = (2,2), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      keras.layers.Dropout(0.25),
      
      keras.layers.SeparableConv2D(64, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      keras.layers.Dropout(0.25),
      
      keras.layers.SeparableConv2D(64, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      keras.layers.Dropout(0.25),
      
      keras.layers.SeparableConv2D(64, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      keras.layers.Dropout(0.25),
      
      keras.layers.SeparableConv2D(128, (3,3), padding = "same"),
      keras.layers.BatchNormalization(),
      keras.layers.Activation("relu"),
      keras.layers.DepthwiseConv2D(kernel_size =(3,3), padding = "same", use_bias = False),
      keras.layers.Dropout(0.25),
      
      keras.layers.MaxPooling2D(pool_size = (4, 4), strides = (4, 4)),
      
      keras.layers.Flatten(),
      
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(32, activation = "relu"),
      keras.layers.Dense(10)
      
      
  ])
 # Add code to define model 1.

  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
    
    
   #(xtrain, ytrain),(xval, yval)
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data() #i think i need one hot encoding for my ytrain and test
   
    
    #xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = .2, random_state = 42)
    
  
    #print(xtrain.shape)
    #print(xtest.shape)
    #print(ytest.shape)
  ########################################

    model1 = build_model1()
    model2 = build_model2()
    model3 = build_model3()    
    model50k = build_model50k()
    
    model1.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model1.summary()


    model2.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model2.summary()


    model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model3.summary()


    model50k.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model50k.summary()


    mdl1 = model1.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=50)

    mdl2 = model2.fit(train_images, train_labels, validation_data = (test_images, test_labels), epochs = 50)

    mdl3 = model3.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=50)

    mdl50k = model50k.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=50)