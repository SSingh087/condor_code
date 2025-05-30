#!/usr/bin/env python

import os
import sys
import argparse
import warnings
import torch
import pickle
import numpy as np

# === Limit threads for libraries ===
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath('../poplar'))

from population import *
from poplar.selection import selection_function_from_optimal_snr

parser = argparse.ArgumentParser()
parser.add_argument("--population", type=str, required=True)
parser.add_argument("--sample_index", type=int, required=True)
parser.add_argument("--work_dir", type=str, required=True)
args = parser.parse_args()

trained_model_path = '/data/wiay/postgrads/shashwat/EMRI_data/trained_models'

# === Load models once ===
with open(f'{trained_model_path}/MODEL_theta_SNR_128_12/best_final_model/model/model.pth', 'rb') as f:
    snr_model = pickle.load(f)
snr_model.eval()

with open(f'{trained_model_path}/MODEL_traj_p0_64_8/best_final_model/model/model.pth', 'rb') as f:
    p0_model = pickle.load(f)
p0_model.eval()

# === Load hyperparams ===
hyperparam_file = f'{args.work_dir}/hyperparams_{args.population}.npy'
hyperparams = np.load(hyperparam_file)
hyperparam = hyperparams[args.sample_index]


weight = torch.tensor(1.0)
# === Select population distribution ===
if args.population == 'A':
    popdist = popdist_A
    true_x = {
        "log10_M": {"xc": torch.tensor(hyperparam[0]).float()},
        "log10_mu": {"mu": torch.tensor(1.5), "sigma": torch.tensor(0.5), "alpha": torch.tensor(hyperparam[1]).float()},
        "a": {"alpha": 12, "beta": 8},
        "e0": {"alpha": 8, "beta": 3},
        "Y0": {}, "dist": {"lam": 3},
        "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
        "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {},
        "T": {}
    }

elif args.population == 'B':
    popdist = popdist_B
    true_x = {
        "log10_M": {"lam": torch.tensor(hyperparam[0]).float()},
        "log10_mu": {"lam": torch.tensor(hyperparam[1]).float()},
        "a": {"mu": torch.tensor(hyperparam[2]).float(), "sigma": torch.tensor(hyperparam[3]).float()},
        "e0": {}, "Y0": {}, "dist": {"lam": 3},
        "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
        "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {},
        "T": {}
    }


elif args.population == 'C':
    popdist = popdist_C
    true_x = {
        "log10_M": {"mu": torch.tensor(hyperparam[0]).float(), "sigma": torch.tensor(3), "alpha": torch.tensor(0.01)},
        "log10_mu": {"mu": torch.tensor(1.2), "sigma": torch.tensor(2.5), "alpha": torch.tensor(hyperparam[1]).float()},
        "a": {"alpha": torch.tensor(hyperparam[2]).float(), "beta": torch.tensor(hyperparam[3]).float()},
        "e0": {"alpha": torch.tensor(hyperparam[4]).float(), "beta": torch.tensor(hyperparam[5]).float()},
        "Y0": {}, "dist": {"lam": 3},
        "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
        "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {},
        "T": {}
    }

elif args.population == 'MIX':
    weight = torch.tensor(hyperparam[10]).float().to(device)
    popdist = popdist_MIX
    true_x = {
        "log10_M": {"lam_A": torch.tensor(hyperparam[0]).float(), "lam_B": torch.tensor(hyperparam[6]).float()},
        "log10_mu": {"mu_A": torch.tensor(1.5), "sigma_A": torch.tensor(0.5), "alpha_A": torch.tensor(hyperparam[1]).float(), "lam_B": torch.tensor(hyperparam[7]).float()},
        "a": {"alpha_A": torch.tensor(hyperparam[2]).float(), "beta_A": torch.tensor(hyperparam[3]).float(), "mu_B": torch.tensor(hyperparam[8]).float(), "sigma_B": torch.tensor(hyperparam[9]).float()},
        "e0": {"alpha_A": torch.tensor(hyperparam[4]).float(), "beta_A": torch.tensor(hyperparam[5]).float(), "UNIFORM_B": {}},
        "Y0": {}, "dist": {"lam": 3},
        "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
        "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {},
        "T": {}
    }

else:
    raise ValueError(f"Unsupported population: {args.population}")

# === Sample catalogue ===
events_per_Lambda = 1000000
snr_th = 20
for _ in range(10000):
    catalogue = popdist.draw_samples(true_x, weight=weight, size=events_per_Lambda)
    if all(not torch.isnan(torch.mean(catalogue[key])) for key in catalogue):
        break

# === Compute selection function ===
p0_s_params = torch.stack([
    catalogue['log10_M'], catalogue['log10_mu'],
    catalogue['a'], catalogue['e0'], catalogue['Y0'], catalogue['T']
], dim=1).float()

with torch.no_grad():
    p0_s = p0_model(p0_s_params)

wave_params = torch.stack([
    catalogue['log10_M'], catalogue['log10_mu'],
    catalogue['a'], p0_s, catalogue['e0'], catalogue['Y0'], catalogue['dist'],
    catalogue['qS'], catalogue['phiS'], catalogue['qK'], catalogue['phiK'],
    catalogue['Phi_phi0'], catalogue['Phi_theta0'], catalogue['Phi_r0'], catalogue['T']
], dim=1).float()

with torch.no_grad():
    catalogue_snrs = snr_model(wave_params)

sf = selection_function_from_optimal_snr(catalogue_snrs, snr_th, number_of_detectors=1)
sf_value = sf.item() if isinstance(sf, torch.Tensor) else sf

# === Save result ===
output_path = os.path.join(f'{args.work_dir}/sf_{args.population}', f'sf_{args.population}_{args.sample_index}.npy')
np.save(output_path, sf_value)
print(f"[INFO] Sample {args.sample_index} complete: sf = {sf_value}")
