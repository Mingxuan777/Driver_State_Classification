# coding=utf-8

# major training and validation file
# load packages
from __future__ import absolute_import, division, print_function
import logging
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from VGG import VGG16, VGG19 # import model from VGG.py
import load_data

import torchvision.models as models
from torch.utils.data import DataLoader


os.environ['KMP_DUPLICATE_LIB_OK']='True' # fix system error in macOS
logger = logging.getLogger(__name__)


def val(model, dataloader):
    '''
    validation function
    '''
    model.eval() # turn model to evaluation mode
    acc_sum = 0 # initiate accuracy sum
    for ii, (input, label) in enumerate(dataloader): # ii is the batch number
        val_input = input
        val_label = label
        if torch.cuda.is_available(): # detect GPU and send data to GPU
            val_input = val_input.cuda()
            val_label = val_label.cuda()

        output = model(val_input) # calculate output result using validation data
        acc_batch = torch.mean(torch.eq(torch.max(output, 1)[1], val_label).float()) # calculate accuracy in a batch
        acc_sum += acc_batch # sum up  batch accuracy

    acc_vali = acc_sum / (ii + 1) # calculate average accuracy per batch
    model.train() # turn model back to train mode
    return acc_vali

def setups():
    '''
    setup parameters
    '''

    parser = argparse.ArgumentParser()
    # Set parameters
    parser.add_argument("--name", default="VGG16", type=str, help="The output directory where logs will be written.")

    parser.add_argument("--model", default="MobileNet", type=str, help="Select MobileNet or VGG16 or VGG19")

    # set up output directory 
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints will be written.")
    # set up image size
    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    # set up epochs --- hyperparameters # 1
    parser.add_argument("--epochs", default=50, type=float, help="Training epoch times")
    # set up learning rate --- hyperparameters # 2                    
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for SGD.")
    # set up batchsize (for training) --- hyperparameters # 3
    parser.add_argument("--train_batch_size", default=32, type=float, help="training batch size.")
    # set up batchsize (for testing) --- hyperparameters # 4
    parser.add_argument("--test_batch_size", default=32, type=float, help="test batch size.")
    # set up seed for reproducibility 
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    # input data path (for training and validation)
    parser.add_argument('--train_data_path', type=str, default='../data1/imgs/train', help="training data path")

    # set up train and validate data size
    parser.add_argument('--traindata_size', type=int, default=0.6, help="training data path")# larger the value, larger the data 
    parser.add_argument('--valdata_size', type=int, default=0.95, help="training data path") # larger the value, smaller the data

    args = parser.parse_args()

    return args

def save_model(model):
    '''
    save trained model
    '''
    model_to_save = model
    model_checkpoint = "./output/model.pt" # model file directory
    torch.save(model_to_save.state_dict(), model_checkpoint) # save model
    logger.info("Saved model checkpoint to [DIR: %s]", model_checkpoint) # record in logs
    # print("model saved !")

def setup(args):
    '''
    setup model 
    '''
    # Prepare model
    # model = VGG16(args) # enable this if you want VGG rather than mobilenet
    if args.model == "MobileNet":
        model = models.mobilenet_v2() # model to use
        model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                        nn.Linear(in_features=model.classifier[1].in_features, out_features=10, bias=True))
        # As mobilenet in torchvision has 1000 classification sets, here an additional fully connected layers are added to make the final 
        # classification sets to 10
    elif args.model == "VGG16":
        model = VGG16()
    elif args.model == "VGG19":
        model = VGG19()

    # detect GPU and send model to GPU
    if torch.cuda.is_available():
        model.cuda()
    
    # count parameters
    num_params = count_parameters(model)

    # write loggers
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model

def count_parameters(model):
    '''
    count model parameters
    '''
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params

def set_seed(args):
    '''
    set seed for reproducibility
    '''
    random.seed(args.seed) 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) # set up pytorch
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, model):
    '''
    Train the model
    '''
    # Prepare dataset
    train_data_path = args.train_data_path

    # read train data, turn shuffle on to shuffle data. num_workers can set to a larger value if multi-GPU is used
    train_data = load_data.LoadDataset(train_data_path, args, train=True)
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=args.train_batch_size, num_workers=0)

    # read validation data
    vali_data = load_data.LoadDataset(train_data_path, args, train=False)
    vali_dataloader = DataLoader(dataset=vali_data, shuffle=False, batch_size=args.train_batch_size, num_workers=0)

    # write tfevent file for tensorboard
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name)) # initiate logger

    # Prepare optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # write logs
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    # print(model) # enable to see model
    model.train() # set model to train

    set_seed(args)  # Added for reproducibility 

    # get epoch, lossfunction and create a new list for containing loss in each run
    epochs = args.epochs
    loss_function = nn.CrossEntropyLoss()
    loss_print = []

    step = 0 # total step number

    # training loop
    for epoch in range(epochs):
        for (input, target) in tqdm(train_dataloader): # use tqdm to visualize  training progress

            optimizer.zero_grad() # clean gradient
            step += 1 # plus 1 to global step
            y = F.one_hot(target, num_classes=10)  # covert labels to one-hot encoding

            # use GPU if you have
            if torch.cuda.is_available():
                input = input.cuda() # pass input data to GPU
                target = y.float().cuda() # pass target data to GPU

            # get out and loss
            out = model(input) # shape of X : (30, 3, 224, 224)
            loss = loss_function(out, target) # calculate MSE loss

            loss.backward()  # gradient descent
            optimizer.step() 
            loss_print.append(loss) # add loss value the list 
            writer.add_scalar("train/loss", scalar_value=loss, global_step = step)

        loss_mean = torch.mean(torch.Tensor(loss_print)) # calculate mean value of losses
        loss_print = [] # clean loss list in this iteration 
        print('The %d th epoch, step : %d' % (epoch, step), 'train_loss: %f' % loss_mean) # print out train loss information
        acc_vali = val(model, vali_dataloader) # calculate validation accuracy
        print('The %d th epoch, acc_vali : %f' % (epoch, acc_vali)) # print out validation accuracy 
        writer.add_scalar("val/acc", scalar_value=acc_vali, global_step=step) # record
        save_model(model) # save the trained model after each epoch

    # save_model(args,model)
    writer.close()
    logger.info("End Training!")
    print("model saved !")

def main():
    args = setups() # set up parameters

    # Setup training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device # assign device parameter

    # Set seed
    set_seed(args)
    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training and validation
    train(args, model)
    
# execute
if __name__ == "__main__":
    main()
