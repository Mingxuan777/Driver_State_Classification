# coding=utf-8
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
from VGG import VGG16, VGG19
from load_data import get_loader

os.environ['KMP_DUPLICATE_LIB_OK']='True' # fix system error in macOS
logger = logging.getLogger(__name__)

def setups():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=False,
                        help="Name of this run. Used for monitoring.",
                        default='new')
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.",
                        required=False)
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--epochs", default=100, type=float,
                        help="Training epoch times")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")

    parser.add_argument("--train_batch_size", default=30, type=float,
                        help="training batch size.")
    parser.add_argument("--test_batch_size", default=30, type=float,
                        help="test batch size.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    return args

def save_model(args, model):
    model_to_save = model.module
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    model = VGG16(args) # define model as DenseNet model, which is densenet.py
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
    
    # Valid

if __name__ == "__main__":
    main()
