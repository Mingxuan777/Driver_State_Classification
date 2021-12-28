# coding=utf-8
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
from VGG import VGG16
import load_data

import torchvision.models as models
from torch.utils.data import DataLoader


os.environ['KMP_DUPLICATE_LIB_OK']='True' # fix system error in macOS
logger = logging.getLogger(__name__)


def val(model, dataloader):
    '''
    validation function
    '''
    model.eval()
    acc_sum = 0
    for ii, (input, label) in enumerate(dataloader):
        val_input = input
        val_label = label
        if torch.cuda.is_available(): # detect GPU
            val_input = val_input.cuda()
            val_label = val_label.cuda()

        output = model(val_input)
        acc_batch = torch.mean(torch.eq(torch.max(output, 1)[1], val_label).float())
        acc_sum += acc_batch

    acc_vali = acc_sum / (ii + 1)
    model.train()
    return acc_vali

def setups():
    '''
    setup parameters
    '''

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default="VGG16", type=str,
                        help="The output directory where checkpoints will be written.")

    #
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--epochs", default=50, type=float,
                        help="Training epoch times")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--train_batch_size", default=32, type=float,
                        help="training batch size.")
    parser.add_argument("--test_batch_size", default=32, type=float,
                        help="test batch size.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--train_data_path', type=str, default='../data1/imgs/train',
                        help="training data path")

    # set up train and validate data size
    parser.add_argument('--traindata_size', type=int, default=0.6, # larger the value, larger the data
                        help="training data path")
    parser.add_argument('--valdata_size', type=int, default=0.95, # larger the value, smaller the data
                        help="training data path")


    args = parser.parse_args()

    return args

def save_model(model):
    '''
    save trained model
    '''
    model_to_save = model
    model_checkpoint = "./output/model.pt"
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", model_checkpoint)
    # print("model saved !")

def setup(args):
    '''
    setup model 
    '''
    # Prepare model
    # model = VGG16(args) # enable this if you want VGG rather than mobilenet
    
    model = models.mobilenet_v2()
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                      nn.Linear(in_features=model.classifier[1].in_features, out_features=10, bias=True))
    
    if torch.cuda.is_available():
        model.cuda()
    num_params = count_parameters(model)

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
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, model):
    '''
    Train the model
    '''
    # Prepare dataset
    train_data_path = args.train_data_path

    # read train data
    train_data = load_data.LoadDataset(train_data_path, args, train=True)
    train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=args.train_batch_size, num_workers=0)

    # read validation data
    vali_data = load_data.LoadDataset(train_data_path, args, train=False)
    vali_dataloader = DataLoader(dataset=vali_data, shuffle=False, batch_size=args.train_batch_size, num_workers=0)

    # write tfevent file for tensorboard
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    # Prepare optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    # model.zero_grad()
    # print(model) # enable to see model
    model.train() # set model to train

    set_seed(args)  # Added for reproducibility 

    # get epoch, lossfunction and create a new list for containing loss in each run
    epochs = args.epochs
    loss_function = nn.CrossEntropyLoss()
    loss_print = []

    step = 0 
    for epoch in range(epochs):
        for (input, target) in tqdm(train_dataloader):

            optimizer.zero_grad()
            step += 1
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
            loss_print.append(loss)
            writer.add_scalar("train/loss", scalar_value=loss, global_step = step)

        loss_mean = torch.mean(torch.Tensor(loss_print))
        loss_print = []
        print('The %d th epoch, step : %d' % (epoch, step), 'train_loss: %f' % loss_mean)
        acc_vali = val(model, vali_dataloader)
        print('The %d th epoch, acc_vali : %f' % (epoch, acc_vali))
        writer.add_scalar("val/acc", scalar_value=acc_vali, global_step=step)
        save_model(model)

    # save_model(args,model)
    writer.close()
    logger.info("End Training!")
    print("model saved !")

def main():
    args = setups()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    set_seed(args)
    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training
    train(args, model)
    # Valid
    # direct_valid(args, model)

if __name__ == "__main__":
    main()
