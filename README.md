# CSC8635 Project

This repository contains CSC 8635 project.

## Basic setups
./logs contains files for tensorboard for visualizing training process.
./output contains model data, which can be used for inference

## To run the analysis
To train neural network, directly run main.py to get the result recored in report.

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

## Dependencies

All dependencies are recoreded in requirements.txt

To install, use the following code:

'''
pip install -r requirements.txt 
'''
