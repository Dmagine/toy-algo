import argparse
import os

import numpy as np
import torch
import torchvision as tv

import models


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--modeldir", type=str)
parser.add_argument("--model", choices=list(models.KNOWN_MODELS.keys()),
                    help="Which variant to use; BiT-M gives best results.")
parser.add_argument("--checkpointdir", type=str)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--heavy", action="store_true", default=False)
parser.add_argument("--sleep_every", type=int, default=0)
args = parser.parse_args()

print(f"Args: {args}")
print("Environ: {}".format({k: v for k, v in os.environ.items()}))

datadir = args.datadir
print(f"Loading dataset from {datadir}")
train_set = tv.datasets.CIFAR100(datadir, train=True, download=False)
valid_set = tv.datasets.CIFAR100(datadir, train=False, download=False)
print(f"Using a training set with {len(train_set)} images.")
print(f"Using a validation set with {len(valid_set)} images.")

modeldir = args.modeldir
model_path = os.path.join(modeldir, f"{args.model}.npz")
print(f"Loading model from {model_path}")
model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes), zero_head=True)
model.load_from(np.load(model_path))
print(f"Loaded model: {model._get_name()}")

checkpointdir = args.checkpointdir
weigth_path = os.path.join(checkpointdir, "weights_resnet.pkl")
print(f"Loading weights from {weigth_path}")
with open(weigth_path, 'rb') as f:
    size = len(f.read())
print(f"Loaded weights: {size}B")
