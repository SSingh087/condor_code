#!/usr/bin/env python

import os, sys, argparse, warnings

sys.path.insert(0, os.path.abspath('../poplar'))

from population import *
from poplar.selection import selection_function_from_optimal_snr
import torch
import pickle

import numpy as np


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--population", type=str, required=True, help="population type")
parser.add_argument("--work_dir", type=str, required=True, help="work_dir")

args = parser.parse_args()

trained_model_path = '/data/wiay/postgrads/shashwat/EMRI_data/trained_models'
output_dir = args.work_dir 

# def load_model(path):
#     try:
#         with open(path, 'rb') as f:
#             model = pickle.load(f)
#         if isinstance(model, torch.nn.Module):
#             print(f"Model loaded from {path}")
#         return model
#     except Exception as e:
#         print(f"Error loading pickle file from {path}: {e}")
#         return None


def init_worker(pop_type, snr_path, p0_path):
    global snr_model, p0_model, population, device, events_per_Lambda, popdist_A, popdist_B, popdist_C, popdist_MIX, snr_th
    import torch
    from poplar.selection import selection_function_from_optimal_snr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(snr_path, 'rb') as f:
        snr_model = pickle.load(f)
    snr_model.eval().to(device)

    with open(p0_path, 'rb') as f:
        p0_model = pickle.load(f)
    p0_model.eval().to(device)

    population = pop_type
    snr_th = 20
    events_per_Lambda = 100000

    from population import popdist_A, popdist_B, popdist_C, popdist_MIX


# load SNR + p0 model
snr_model_path = f'{trained_model_path}/MODEL_theta_SNR_128_12/best_final_model/model/model.pth'
p0_model_path = f'{trained_model_path}/MODEL_traj_p0_64_8/best_final_model/model/model.pth'

# def eval_sf_s(num_lambda, hyperparams):
    
#     weight = torch.tensor(1.0)

#     if args.population == 'A':

#         true_x = {
            
#                 "log10_M": {
#                         "xc": torch.as_tensor(hyperparams[0]).float()
#                         },

#                 "log10_mu": {
#                         "mu": torch.tensor(1.5),
#                         "sigma": torch.tensor(0.5),
#                         "alpha": torch.as_tensor(hyperparams[1]).float()
#                         },

#                 # "a": {
#                 #         "alpha": torch.as_tensor(hyperparams[2]).float(),
#                 #         "beta": torch.as_tensor(hyperparams[3]).float()
#                 #     },

#                 # "e0": {
#                 #         "alpha": torch.as_tensor(hyperparams[4]).float(),
#                 #         "beta": torch.as_tensor(hyperparams[5]).float()
#                 #     },

#                 "a": {"alpha": 12, "beta": 8},
#                 "e0": {"alpha": 8, "beta": 3},
#                 "Y0": {},
#                 "dist": {"lam": 3},
#                 "qS": {},
#                 "phiS": {},
#                 "qK": {},
#                 "phiK": {},
#                 "Phi_phi0": {},
#                 "Phi_theta0": {},
#                 "Phi_r0": {},
#                 "T" : {},
#                 }

#         # if not os.path.exists(f'{output_dir}/data_A/'):
#         #     os.makedirs(f'{output_dir}/data_A/', exist_ok=True)

#         popdist = popdist_A

#     if args.population == 'B':

#         true_x = {
            
#                 "log10_M": {
#                         "lam": torch.as_tensor(hyperparams[0]).float()
#                         },

#                 "log10_mu": {
#                         "lam": torch.as_tensor(hyperparams[1]).float()
#                         },

#                 "a": {
#                         "mu": torch.as_tensor(hyperparams[2]).float(),
#                         "sigma": torch.as_tensor(hyperparams[3]).float()
#                     },

#                 "e0": {},
#                 "Y0": {},
#                 "dist": {"lam": 3},
#                 "qS": {},
#                 "phiS": {},
#                 "qK": {},
#                 "phiK": {},
#                 "Phi_phi0": {},
#                 "Phi_theta0": {},
#                 "Phi_r0": {},
#                 "T" : {},
#                 }
        
#         # if not os.path.exists(f'{output_dir}/data_B/'):
#         #     os.makedirs(f'{output_dir}/data_B/', exist_ok=True)

#         popdist = popdist_B
 
#     if args.population == 'C':

#         true_x = {
            
#                 "log10_M": {
#                         "mu": torch.as_tensor(hyperparams[0]).float(),
#                         "sigma": torch.tensor(3),
#                         "alpha": torch.tensor(0.01)
#                         },

#                 "log10_mu": {
#                         "mu": torch.tensor(1.2),
#                         "sigma": torch.tensor(2.5),
#                         "alpha": torch.as_tensor(hyperparams[1]).float()
#                         },

#                 "a": {
#                         "alpha": torch.as_tensor(hyperparams[2]).float(),
#                         "beta": torch.as_tensor(hyperparams[3]).float()
#                     },

#                 "e0": {
#                         "alpha": torch.as_tensor(hyperparams[4]).float(),
#                         "beta": torch.as_tensor(hyperparams[5]).float()
#                     },

#                 "Y0": {},
#                 "dist": {"lam": 3},
#                 "qS": {},
#                 "phiS": {},
#                 "qK": {},
#                 "phiK": {},
#                 "Phi_phi0": {},
#                 "Phi_theta0": {},
#                 "Phi_r0": {},
#                 "T" : {},
#                 }

#         # if not os.path.exists(f'{output_dir}/data_C/'):
#         #     os.makedirs(f'{output_dir}/data_C/', exist_ok=True)

#         popdist = popdist_C

#     if args.population == 'MIX':

#         true_x = {
#                 "log10_M": {
#                         "lam_A": torch.as_tensor(hyperparams[0]).float(),
#                          "lam_B": torch.as_tensor(hyperparams[6]).float()
#                         },

#                 "log10_mu": {
#                         "mu_A": torch.tensor(1.5), "sigma_A": torch.tensor(0.5), "alpha_A": torch.as_tensor(hyperparams[1]).float(), 
#                         "lam_B": torch.as_tensor(hyperparams[7]).float()
#                         },

#                 "a": {
#                         "alpha_A": torch.as_tensor(hyperparams[2]).float(), "beta_A": torch.as_tensor(hyperparams[3]).float(),
#                         "mu_B": torch.as_tensor(hyperparams[8]).float(), "sigma_B": torch.as_tensor(hyperparams[9]).float()
#                     },

#                 "e0": {
#                         "alpha_A": torch.as_tensor(hyperparams[4]).float(), "beta_A": torch.as_tensor(hyperparams[5]).float(),
#                         "UNIFORM_B" : {}
#                     },

#                 "Y0": {},
#                 "dist": {"lam": 3},
#                 "qS": {},
#                 "phiS": {},
#                 "qK": {},
#                 "phiK": {},
#                 "Phi_phi0": {},
#                 "Phi_theta0": {},
#                 "Phi_r0": {},
#                 "T" : {},
#                 }


#         weight = torch.as_tensor(hyperparams[10]).float()
        
#         # if not os.path.exists(f'{output_dir}/data_MIX/'):
#         #     os.makedirs(f'{output_dir}/data_MIX/', exist_ok=True)
        
#         popdist = popdist_MIX

#     ## generating events according to the distribution 

#     check = 0
#     while check < 10000:
#         catalogue = popdist.draw_samples(true_x, weight=weight, size=int(events_per_Lambda))

#         # Assume all means are valid until proven otherwise
#         all_means_valid = True

#         for key in catalogue:
#             if torch.isnan(torch.mean(catalogue[key])):
#                 all_means_valid = False
#                 break

#         if all_means_valid:
#             break

#         print(f"CHECK : {check}")

#         check += 1


#     p0_s_params = np.array([
#                             catalogue['log10_M'], catalogue['log10_mu'],
#                             catalogue['a'], catalogue['e0'], catalogue['Y0'],
#                             catalogue['T']
#                             ], dtype=np.float32).T

#     p0_s = p0_model.run_on_dataset(torch.from_numpy(p0_s_params))

#     if torch.isnan(torch.mean(p0_s)):
#         breakpoint()

#     ## save these parameters to check against the true SNR from the waveform     
#     # dtype=np.float32 is required because p0_s stores values in float64. 
#     # Since we have given get_separatrix(np.float64(a.numpy()), np.float64(e0.numpy()), np.float64(Y0.numpy()))
#     # if we were to give float32 precision would have decreased.
#     # so we caluclate p0 with float64 and then downcast it to float32 for 
#     # to match the dtype of the trained model input

#     wave_params = np.array([
#                             catalogue['log10_M'], catalogue['log10_mu'],
#                             catalogue['a'], p0_s, catalogue['e0'], catalogue['Y0'], catalogue['dist'],
#                             catalogue['qS'], catalogue['phiS'], catalogue['qK'], catalogue['phiK'],
#                             catalogue['Phi_phi0'], catalogue['Phi_theta0'], catalogue['Phi_r0'], 
#                             catalogue['T']
#                            ], dtype=np.float32).T
    
#     if np.isnan(np.mean(wave_params)):
#         breakpoint()



#     # np.save(f'{output_dir}/data_{args.population}/params_{num_lambda}.npy', wave_params)

#     ## evaluate SNR for the parameters using the trained SNR model
#     ## sequence should be same as that is trained on 
#     catalogue_snrs = snr_model.run_on_dataset(torch.from_numpy(wave_params))
#     # print(torch.mean(catalogue_snrs))

#     ## check for NaN values in SNRs 
#     if torch.isnan(catalogue_snrs).any():
#         print("contain NaN values")

#     # np.save(f'{output_dir}/data_{args.population}/snrs_{num_lambda}.npy', catalogue_snrs)

#     sf = selection_function_from_optimal_snr(catalogue_snrs, snr_th, number_of_detectors=1)

#     ## evaluate selection function see Magorrie chap -7 
#     return sf


def eval_sf_s(args_tuple):
    i, hyperparams = args_tuple

    weight = torch.tensor(1.0).to(device)

    if population == 'A':
        true_x = {
            "log10_M": {"xc": torch.as_tensor(hyperparams[0]).float()},
            "log10_mu": {"mu": torch.tensor(1.5), "sigma": torch.tensor(0.5), "alpha": torch.as_tensor(hyperparams[1]).float()},
            "a": {"alpha": 12, "beta": 8},
            "e0": {"alpha": 8, "beta": 3},
            "Y0": {}, "dist": {"lam": 3},
            "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
            "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {},
            "T": {}
        }
        popdist = popdist_A

    elif population == 'B':
        true_x = {
            "log10_M": {"lam": torch.as_tensor(hyperparams[0]).float()},
            "log10_mu": {"lam": torch.as_tensor(hyperparams[1]).float()},
            "a": {"mu": torch.as_tensor(hyperparams[2]).float(), "sigma": torch.as_tensor(hyperparams[3]).float()},
            "e0": {}, "Y0": {}, "dist": {"lam": 3},
            "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
            "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {},
            "T": {}
        }
        popdist = popdist_B

    elif population == 'C':
        true_x = {
            "log10_M": {"mu": torch.as_tensor(hyperparams[0]).float(), "sigma": torch.tensor(3), "alpha": torch.tensor(0.01)},
            "log10_mu": {"mu": torch.tensor(1.2), "sigma": torch.tensor(2.5), "alpha": torch.as_tensor(hyperparams[1]).float()},
            "a": {"alpha": torch.as_tensor(hyperparams[2]).float(), "beta": torch.as_tensor(hyperparams[3]).float()},
            "e0": {"alpha": torch.as_tensor(hyperparams[4]).float(), "beta": torch.as_tensor(hyperparams[5]).float()},
            "Y0": {}, "dist": {"lam": 3},
            "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
            "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {},
            "T": {}
        }
        popdist = popdist_C

    elif population == 'MIX':
        true_x = {
            "log10_M": {"lam_A": torch.as_tensor(hyperparams[0]).float(), "lam_B": torch.as_tensor(hyperparams[6]).float()},
            "log10_mu": {"mu_A": torch.tensor(1.5), "sigma_A": torch.tensor(0.5), "alpha_A": torch.as_tensor(hyperparams[1]).float(), "lam_B": torch.as_tensor(hyperparams[7]).float()},
            "a": {"alpha_A": torch.as_tensor(hyperparams[2]).float(), "beta_A": torch.as_tensor(hyperparams[3]).float(), "mu_B": torch.as_tensor(hyperparams[8]).float(), "sigma_B": torch.as_tensor(hyperparams[9]).float()},
            "e0": {"alpha_A": torch.as_tensor(hyperparams[4]).float(), "beta_A": torch.as_tensor(hyperparams[5]).float(), "UNIFORM_B": {}},
            "Y0": {}, "dist": {"lam": 3},
            "qS": {}, "phiS": {}, "qK": {}, "phiK": {},
            "Phi_phi0": {}, "Phi_theta0": {}, "Phi_r0": {},
            "T": {}
        }
        weight = torch.as_tensor(hyperparams[10]).float().to(device)
        popdist = popdist_MIX

    # Draw valid samples
    check = 0
    while check < 10000:
        catalogue = popdist.draw_samples(true_x, weight=weight, size=int(events_per_Lambda))
        if all(not torch.isnan(torch.mean(catalogue[key])) for key in catalogue):
            break
        check += 1

    p0_s_params = np.array([
        catalogue['log10_M'], catalogue['log10_mu'],
        catalogue['a'], catalogue['e0'], catalogue['Y0'], catalogue['T']
    ], dtype=np.float32).T

    with torch.no_grad():
        p0_s = p0_model(torch.from_numpy(p0_s_params).to(device))

    wave_params = np.array([
        catalogue['log10_M'], catalogue['log10_mu'],
        catalogue['a'], p0_s.cpu().numpy(), catalogue['e0'], catalogue['Y0'], catalogue['dist'],
        catalogue['qS'], catalogue['phiS'], catalogue['qK'], catalogue['phiK'],
        catalogue['Phi_phi0'], catalogue['Phi_theta0'], catalogue['Phi_r0'], catalogue['T']
    ], dtype=np.float32).T

    with torch.no_grad():
        catalogue_snrs = snr_model(torch.from_numpy(wave_params).to(device))

    if torch.isnan(catalogue_snrs).any():
        print(f"NaNs in SNRs for sample {i}")

    sf = selection_function_from_optimal_snr(catalogue_snrs, snr_th, number_of_detectors=1)

    return sf.item() if isinstance(sf, torch.Tensor) else sf


# if __name__ == "__main__":

#     snr_model = load_model(snr_model_path)
#     p0_model = load_model(p0_model_path)

#     if args.population == "A":
#         hyperparams = np.load(f'{output_dir}/hyperparams_A.npy')

#     elif args.population == "B":
#         hyperparams = np.load(f'{output_dir}/hyperparams_B.npy')

#     elif args.population == "C":
#         hyperparams = np.load(f'{output_dir}/hyperparams_C.npy')

#     elif args.population == "MIX":
#         hyperparams = np.load(f'{output_dir}/hyperparams_MIX.npy')
    
#     # SNR THRESHOLD 
#     snr_th = 20

#     sf = []
#     for i in range(len(hyperparams)):
#         sf.append(eval_sf_s(i, hyperparams[i]))

#     np.save(f'{output_dir}/sf_{args.population}.npy', np.array(sf))


#     ## Plotting data 
#     import matplotlib.pyplot as plt
#     plt.hist(np.array(sf), bins='auto')
#     plt.savefig('sf_A.png')

#     # RUNNING WITH MULTIPROCESS IS NOT RECOMMENDED WITH PYTORCH
#     # from multiprocessing import Pool
#     # pool = Pool(32)
#     # sf = pool.map(eval_sf_s, range(len(lambda_s)))
#     # pool.close()  # Close the pool to prevent any more tasks from being submitted
#     # pool.join()   # Wait for the worker processes to exit


if __name__ == "__main__":
    import multiprocessing as mp

    # Set paths
    snr_model_path = '/data/wiay/postgrads/shashwat/EMRI_data/trained_models/MODEL_theta_SNR_128_12/best_final_model/model/model.pth'
    p0_model_path = '/data/wiay/postgrads/shashwat/EMRI_data/trained_models/MODEL_traj_p0_64_8/best_final_model/model/model.pth'

    # Set number of workers (adjust for your system)
    num_workers = mp.cpu_count()

    # Load hyperparameters based on population
    if args.population == "A":
        hyperparams = np.load(f'{args.work_dir}/hyperparams_A.npy')
    elif args.population == "B":
        hyperparams = np.load(f'{args.work_dir}/hyperparams_B.npy')
    elif args.population == "C":
        hyperparams = np.load(f'{args.work_dir}/hyperparams_C.npy')
    elif args.population == "MIX":
        hyperparams = np.load(f'{args.work_dir}/hyperparams_MIX.npy')
    else:
        raise ValueError(f"Unknown population type: {args.population}")

    # Create list of (index, hyperparam) tuples for mapping
    param_list = [(i, hyperparams[i]) for i in range(len(hyperparams))]

    # Run multiprocessing pool
    with mp.get_context("spawn").Pool(processes=num_workers, initializer=init_worker,
                                      initargs=(args.population, snr_model_path, p0_model_path)) as pool:
        sf_values = pool.map(eval_sf_s, param_list)

    # Save results
    np.save(f'{args.work_dir}/sf_{args.population}.npy', np.array(sf_values))

    # Plot
    import matplotlib.pyplot as plt
    plt.hist(sf_values, bins='auto')
    plt.savefig(f'sf_{args.population}_hist.png')
