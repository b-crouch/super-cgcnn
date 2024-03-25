import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from data import *
from model import CrystalGraphConvNet
from utils import *

import matplotlib.pyplot as plt
import warnings
import os
import datetime
import csv
from tqdm import tqdm
import pandas as pd

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options: path to directory of CIF data, filepath to ground truth labels, # graph neighbors, neighbor radius')
parser.add_argument("--struct_features", action="store_true", 
                    help="If passed, network will accept non-elemental features as input")
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR', 
                    help='initial learning rate (default: 1e-5)')
parser.add_argument('--lr_milestones', default=[100], nargs='+', type=int, metavar='N', 
                    help='milestones for learning rate scheduler (default: [100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train_ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train_size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')

valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val_ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val_size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')

test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test_ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test_size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n_fc', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

parser.add_argument("--title", default="vanilla_cgcnn", type=str,
                    help="title of experiment, used for naming output files")

parser.add_argument("--pool_func", default="mean", type=str, choices=["mean", "min", "max", "l2"])
parser.add_argument("--activation", default="softplus", type=str, choices=["softplus", "relu", "leaky_relu"])

parser.add_argument("--fc_dim", type=int, nargs="+",
                    help="dimensionality of post-convolution fully-connected layers")

args = parser.parse_args(sys.argv[1:])

if torch.cuda.is_available():
    device = torch.device("cuda")
    args.cuda = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    args.cuda = True
else:
    device = torch.device("cpu")
    args.cuda = False

assert len(args.fc_dim)==args.n_fc, "Number of hidden layers must match number of hidden layer dimensions"

pooling_map = {"mean":torch.mean, "min":torch.min, "max":torch.max, "l2":torch.norm}
activation_map = {"softplus":nn.Softplus, "relu":nn.ReLU, "leaky_relu":nn.LeakyReLU}

best_mae_error = 0

start = datetime.datetime.now()
print(f"Begin loading data: {start}")

dataset = CIFData(*args.data_options)
collate_fn = collate_pool
train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset=dataset,
    collate_fn=collate_fn,
    batch_size=args.batch_size,
    train_ratio=args.train_ratio,
    num_workers=args.workers,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    pin_memory=args.cuda,
    train_size=args.train_size,
    val_size=args.val_size,
    test_size=args.test_size,
    return_test=True)

normalizer = Normalizer(torch.zeros(2))
normalizer.load_state_dict({'mean': 0., 'std': 1.})

print(f"Begin building model: {datetime.datetime.now()}")
structures, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]
structure_fea_len = structures[3].shape[-1] if structures[3] is not None else 0

model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, structure_fea_len,
                            atom_fea_len=args.atom_fea_len,
                            n_conv=args.n_conv,
                            n_fc=args.n_fc,
                            fc_dim=args.fc_dim,
                            activation=activation_map[args.activation],
                            classification=True)

model.to(device)

criterion = nn.NLLLoss()

if args.optim == 'SGD':
    optimizer = optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(model.parameters(), args.lr,
                            weight_decay=args.weight_decay)
else:
    raise NameError('Only SGD or Adam is allowed as --optim')

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_mae_error = checkpoint['best_mae_error']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

print(f"Begin training: {datetime.datetime.now()}")
train_scores = np.zeros(args.epochs)
val_scores = np.zeros(args.epochs)
num_iter = (args.epochs-args.start_epoch)*len(train_loader)

train_losses = AverageMeter()
train_auc_scores = AverageMeter()
val_losses = AverageMeter()
val_auc_scores = AverageMeter()
auc_history = np.zeros(args.epochs)
train_history, val_history = np.zeros(args.epochs), np.zeros(args.epochs)

with tqdm(total=num_iter) as progress_bar:
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        for i, (input, target, batch_cif_ids) in enumerate(train_loader):
            structure_features = input[3]
            if structure_features is not None:
                structure_features = structure_features.to(device)
            input_var = (Variable(input[0].to(device)),
                            Variable(input[1].to(device)),
                            input[2].to(device),
                            structure_features,
                            [crys_idx.to(device) for crys_idx in input[4]])
            target_normed = target.view(-1).long()
            target_var = Variable(target_normed.to(device))
            output = model(*input_var, pool_func=pooling_map[args.pool_func])
            loss = criterion(output, target_var)
            accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
            train_losses.update(loss.data.cpu().item(), target.size(0))
            train_auc_scores.update(auc_score, target.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
        print("-"*30)
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(fr"** Training loss: {np.round(train_losses.avg, 5)}")
        print(fr"** Training AUC: {np.round(train_auc_scores.avg, 5)}")
        train_history[epoch] = train_losses.avg
        train_losses.reset()
        train_auc_scores.reset()

        with torch.no_grad():
            for i, (input, target, batch_cif_ids) in enumerate(val_loader):
                structure_features = input[3]
                if structure_features is not None:
                    structure_features = structure_features.to(device)
                input_var = (Variable(input[0].to(device)),
                                Variable(input[1].to(device)),
                                input[2].to(device),
                                structure_features,
                                [crys_idx.to(device) for crys_idx in input[4]])
                target_normed = target.view(-1).long()
                target_var = Variable(target_normed.to(device))
                output = model(*input_var, pool_func=pooling_map[args.pool_func])
                loss = criterion(output, target_var)
                accuracy, precision, recall, fscore, auc_score = \
                        class_eval(output.data.cpu(), target)
                val_losses.update(loss.data.cpu().item(), target.size(0))
                val_auc_scores.update(auc_score, target.size(0))
            print(fr"** Val loss: {np.round(val_losses.avg, 5)}")
            print(fr"** Val AUC: {np.round(val_auc_scores.avg, 5)}")
            auc_history[epoch] = val_auc_scores.avg
            val_history[epoch] = val_losses.avg
            val_losses.reset()
            val_auc_scores.reset()

pd.DataFrame({"train_loss":train_history, "val_loss":val_history, "auc":auc_history}).to_csv(f"{args.title}_results")