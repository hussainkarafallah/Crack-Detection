#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import DenseNet121


image_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = preprocess_input ,
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 45,
    validation_split = 0.2
)

train_gen = image_gen.flow_from_directory(
    './Dataset/' ,
    batch_size = 32 ,
    target_size = (128 , 128),
    classes = ['Negative','Positive'] ,class_mode = 'binary' ,
    shuffle = True , subset = 'training'
)

val_gen = image_gen.flow_from_directory(
    './Dataset/' , 
    batch_size = 32 ,
    target_size = (128 , 128),
    classes = ['Negative','Positive'] ,class_mode = 'binary' , 
    shuffle = True , subset = 'validation'
)


# In[8]:


densenet = DenseNet121(include_top=False, weights='imagenet', input_shape = (128,128,3))
densenet.trainable = False
embeddings = densenet.get_layer('pool4_relu').output
flat = keras.layers.Flatten(name = "flatten")(embeddings)
dense = keras.layers.Dense(units = 128 , activation = 'relu')(flat)
output = keras.layers.Dense(units = 1 , activation = 'sigmoid')(dense)
model = keras.Model(inputs = densenet.input , outputs = output)
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])


# In[9]:


model.summary()


# In[10]:


early = keras.callbacks.EarlyStopping(monitor = 'val_accuracy' , patience = 1 , mode = 'max' , min_delta = 0.01)
model.fit(train_gen , epochs = 100 , validation_data = val_gen , callbacks = [early])


# In[12]:


model.save("trained.h5")


# In[13]:


model.summary()

