from trainer import Trainer
import argparse

import torch
import numpy
import random

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
numpy.random.seed(42)
random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument("--type", default='gan')
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--diter", default=5, type=int)
parser.add_argument("--save_path", default='')
parser.add_argument('--pre_trained_disc', default=None)
parser.add_argument('--pre_trained_gen', default=None)
parser.add_argument('--dataset', default='flowers')
parser.add_argument('--split', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--epochs', default=200, type=int)

args = parser.parse_args()

trainer = Trainer(**vars(args))

trainer.train(False)

# trainer.predict()
