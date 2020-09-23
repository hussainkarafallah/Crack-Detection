# Crack Detection - Hussain Kara Fallah

# IMPORTANT INSTRUCTIONS:

`train.py` This script is for training make sure that all the photos are inside `Dataset` Directory the same as the one you have provided and it should train.

Because training takes a bit of time I already generated the model itself and saved it (seralized h5) so that you can do prediction without the need of training again. However feel free to run the training again.

`main.py` this is the file for generating predictions you can call it like `python3 main.py ./test/` and it will generate predcitions for images which are in side `/test/` directory (or other directory of your choice). The images should be inside the folder directly with no sub-directories.

`trained.zip` This archive contains the trained model. Unzip it after downloading please. After unzipping you will have the file:

`trained.h5` This is the model I trained on my machine

`predictions.csv` This is just a sample corresponding to the photos in `test` directory



## Requirements:

* tensorflow 2.0
* pandas
* matplotlib
* sklearn