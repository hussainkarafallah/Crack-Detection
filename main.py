import argparse
import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import DenseNet121

def solve(test_path):
    # checking filenames in test dir
    filenames = sorted(os.listdir(test_path))
    
    # reading images
    images = [ load_img(os.path.join(test_path , f) , target_size = (128,128))  for f in filenames ]
    images = [ img_to_array (x , dtype = np.float32) for x in images ]
    # stacking them along first axis (N,H,W,C)
    images = np.vstack([np.expand_dims(x , 0) for x in images]).astype(np.float32)
    # preprocessing
    images = preprocess_input(images)

    # loading + prediction
    model = tf.keras.models.load_model("./trained.h5" , compile = False)
    predictions = model.predict(images)
    predictions = (predictions >= 0.5).astype(int)

    # saving results
    results = pd.DataFrame( {'file' : filenames , 'label' : predictions.squeeze()})
    results.to_csv("predictions.csv" , index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path" , action='store' , type = str , help = 'path to the testing images')
    args = parser.parse_args()
    solve(args.test_path)



