#!/usr/bin/env python
# coding: utf-8




import tensorflow as tf
import scipy 
from tensorflow import keras
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score




physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", physical_devices)



from tensorflow.keras import backend as K

#recall metric
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#precision metric
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#f1-score metric
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))





from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import DenseNet121


#image data generator for processing image + doing simple augmentations (horizontal/vertical flip + rotiation)
image_gen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = preprocess_input ,
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 45,
    validation_split = 0.2
)

#training_data generator to flow from directory
train_gen = image_gen.flow_from_directory(
    './Dataset/' ,
    batch_size = 32 ,
    target_size = (128 , 128),
    classes = ['Negative','Positive'] ,class_mode = 'binary' ,
    shuffle = True , subset = 'training'
)

#validation data generator
val_gen = image_gen.flow_from_directory(
    './Dataset/' , 
    batch_size = 32 ,
    target_size = (128 , 128),
    classes = ['Negative','Positive'] ,class_mode = 'binary' , 
    shuffle = True , subset = 'validation'
)



#Transfer learning model (densenet) + Freezing
densenet = DenseNet121(include_top=False, weights='imagenet', input_shape = (128,128,3))
densenet.trainable = False
#getting embeddings from pretrained densenet
embeddings = densenet.get_layer('pool4_relu').output
#fine tuning by adding 2 dense layers
flat = keras.layers.Flatten(name = "flatten")(embeddings)
dense = keras.layers.Dense(units = 128 , activation = 'relu')(flat)
output = keras.layers.Dense(units = 1 , activation = 'sigmoid')(dense)
model = keras.Model(inputs = densenet.input , outputs = output)
#compiling model
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy',f1_m,precision_m,recall_m])




model.summary()



# fitting with early stopping on validation data, early stopping on f1 metric
early = keras.callbacks.EarlyStopping(monitor = 'val_f1_m' , patience = 1 , mode = 'max' , min_delta = 0.01)
model.fit(train_gen , epochs = 100 , validation_data = val_gen , callbacks = [early])




model.save("trained.h5")




