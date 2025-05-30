#!/usr/bin/env python

import os, sys, argparse, warnings
import numpy as np
import matplotlib.pyplot as plt
import torch, pickle
sys.path.insert(0, os.path.abspath('../poplar'))
from scipy.stats import gaussian_kde


sf_dir = '/data/wiay/postgrads/shashwat/EMRI_data/SF_MODEL_TEST'

hyperparams_A = np.load(f'{sf_dir}/hyperparams_A_test.npy')
sf_A = np.load(f'{sf_dir}/sf_A_test.npy')

print(hyperparams_A.shape, sf_A.shape)

with open(f'{sf_dir}/MODEL_hyperparam_SF_A_256_14_64/best_final_model/model/model.pth', 'rb') as f:
    sf_model = pickle.load(f)
sf_model.eval()

breakpoint()
sf_trained = []

for i in range(len(sf_A)):

    sf_trained.append(sf_model.run_on_dataset(
                                                torch.tensor(hyperparams_A[i,0]), 
                                                torch.tensor(hyperparams_A[i,1])
                                                )
                    )

plt.hist(sf_A, bins='auto', label='generated')
plt.hist(np.array(sf_trained), bins='auto', label='trained')
plt.legend()
plt.savefig('sf_A_comp.png')