# CSC8635 Project

This repository contains CSC 8635 project.

## Basic setups
./logs contains files for tensorboard for visualizing training process.
./output contains model data, which can be used for inference

Data are stored in ../data file, which is excluded in this git repository to save file size. 

## To run the analysis
To train neural network, directly run main.py to get the result recored in report.
To change default model, see the following to choose VGG16, VGG19 or MobileNet

## Options to run (hyperparameters and setup)
Since argparse is implemented, it is able to use additional argument to change code settings:

$ python3 main.py 

add *--output_dir=value* to change output directory
add *--epoch=value* to change epoch size
add *--learning_rate=value* to change learning rate
add *--train_batch_size=size* to change training batch size
add *--train_data_path=path* to change training data batch path

add *--traindata_size=value* to change training data size
add *--valdata_size=value* to change validation data size
add *--model=VGG16, VGG19, MobileNet* to change training neural network

## Dependencies

All dependencies are recoreded in requirements.txt

To install, use the following code:

'''
pip install -r requirements.txt 
'''
