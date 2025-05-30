#!/usr/bin/env python

import os, sys
sys.path.insert(0, os.path.abspath('../poplar'))
from poplar.nn.networks import LinearModel
from poplar.nn.training import train, train_test_split
from poplar.nn.rescaling import ZScoreRescaler
import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--x_data_loc", type=str, required=True)
parser.add_argument("--y_data_loc", type=str, required=True)
parser.add_argument("--train_cat", type=str, required=True)
parser.add_argument("--num_neurons", type=int, required=True)
parser.add_argument("--layers", type=int, required=True)
parser.add_argument("--train_test_frac", type=float, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--n_epochs", type=int, required=True)
parser.add_argument("--n_batches", type=int, required=True)
parser.add_argument("--update_every", type=int, required=True)
parser.add_argument("--verbose", type=bool, required=True)
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--device", type=str, required=True)

args = parser.parse_args()

output_dir = f'/data/wiay/postgrads/shashwat/EMRI_data/SF_MODEL_TEST/'

device = args.device

# load x data
x = np.load(args.x_data_loc)

# load y data
y = np.load(args.y_data_loc)

xdata = torch.as_tensor(x, device=device).float()
ydata = torch.as_tensor(y, device=device).float() 

print("Input Shape : ", xdata.shape, "Output shape : ", ydata.shape)

if args.train_cat == 'SNR':
    rescaler = ZScoreRescaler(xdata, ydata)#, yfunctions=[torch.log, torch.exp])
elif args.train_cat == 'SF':
    rescaler = ZScoreRescaler(xdata, ydata) 
elif args.train_cat == 'p0':
    rescaler = ZScoreRescaler(xdata, ydata) 
else :
    raise KeyError("only SNR, SF and p0 are the allowed keywords")

model = LinearModel(
    in_features=xdata.shape[-1],
    out_features=1,
    neurons=[args.num_neurons] * args.layers,
    activation=torch.nn.SiLU,
    rescaler=rescaler,
    # dropout=0.1
)

# no shuffling is required since this is random data 

model.set_device(device)

xtrain, xtest, ytrain, ytest = train_test_split([xdata, ydata], args.train_test_frac)  
optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience=1, factor=0.5)


train(
    model, 
    data=[xtrain, ytrain, xtest, ytest], 
    n_epochs=args.n_epochs, 
    n_batches=args.n_batches, 
    loss_function=torch.nn.L1Loss(),
    optimiser=optimiser,
    update_every=args.update_every,
    verbose=args.verbose,
    save_best=True,
    scheduler=scheduler,
    outdir=f'{output_dir}/{args.outdir}_{args.num_neurons}_{args.layers}_{args.n_batches}/',
)

# check errors on the model

ypred = model.run_on_dataset(xtest)

plt.hist(np.log10(abs((1 - ypred/ytest).cpu().numpy())), bins='auto', density=True)
plt.xlabel('log_10 percent error')
plt.savefig(f'{output_dir}/{args.outdir}_{args.num_neurons}_{args.layers}_{args.n_batches}/percent_error.png', dpi=200)
plt.close()