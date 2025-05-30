#!/usr/bin/env python

import os, sys
sys.path.insert(0, os.path.abspath('../poplar'))
from poplar.distributions import *
from numpy import pi, arccos
import re

device = "cpu"

# This is the new population that captures current know physics 
# but this is only a phenomenological model
distributions_A = {
    "log10_M": FixedLimitSchechterFunction([6, 10], device=device),
    "log10_mu": FixedLimitsTruncatedSkewNormal([1, 2], device=device),
    "a" : FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device),
    "e0" : FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device),
    "Y0": UniformDistribution([0.1, 0.7], device=device),
    "dist": FixedLimitsPowerLaw([1, 8], device=device),
    "qS": UniformDistribution([0.1, pi*0.99], device=device),
    "phiS": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "qK": UniformDistribution([0.1, pi*0.99], device=device),
    "phiK": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_phi0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_theta0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_r0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "T" : UniformDistribution([1, 6], device=device),
}

# this population is derived from https://doi.org/10.1093/mnras/stad1397
distributions_B = {
    "log10_M": FixedLimitsPowerLaw([6, 10], device=device),
    "log10_mu": FixedLimitsPowerLaw([1, 2], device=device),
    "a" : FixedLimitsTruncatedGaussian([0.1, 0.7], device=device),
    "e0" : UniformDistribution([0.1, 0.7], device=device),
    "Y0": UniformDistribution([0.1, 0.7], device=device),
    "dist": FixedLimitsPowerLaw([1, 8], device=device),
    "qS": UniformDistribution([0.1, pi*0.99], device=device),
    "phiS": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "qK": UniformDistribution([0.1, pi*0.99], device=device),
    "phiK": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_phi0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_theta0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_r0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "T" : UniformDistribution([1, 6], device=device),
}

# This is the new population that captures current know physics 
# but this is only a phenomenological model
distributions_C = {
    "log10_M": FixedLimitsTruncatedSkewNormal([6, 10], device=device),
    "log10_mu": FixedLimitsTruncatedSkewNormal([1, 2], device=device),
    "a" : FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device),
    "e0" : FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device),
    "Y0": UniformDistribution([0.1, 0.7], device=device),
    "dist": FixedLimitsPowerLaw([1, 8], device=device),
    "qS": UniformDistribution([0.1, pi*0.99], device=device),
    "phiS": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "qK": UniformDistribution([0.1, pi*0.99], device=device),
    "phiK": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_phi0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_theta0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_r0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "T" : UniformDistribution([1, 6], device=device),
}

# A population is derived from https://doi.org/10.1093/mnras/stad1397
# B population is new
POLULATION_MIX = {
    "log10_M": [FixedLimitSchechterFunction([6, 10], device=device), FixedLimitsPowerLaw([6, 10], device=device)],
    "log10_mu": [FixedLimitsTruncatedSkewNormal([1, 2], device=device), FixedLimitsPowerLaw([1, 2], device=device)],
    "a" : [FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device), FixedLimitsTruncatedGaussian([0.1, 0.7], device=device)],
    "e0" : [FixedLimitTruncatedBetaDistribution([0.1, 0.7], device=device), UniformDistribution([0.1, 0.7], device=device)],
    "Y0": UniformDistribution([0.1, 0.7], device=device),
    "dist": FixedLimitsPowerLaw([1, 8], device=device),
    "qS": UniformDistribution([0.1, pi*0.99], device=device),
    "phiS": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "qK": UniformDistribution([0.1, pi*0.99], device=device),
    "phiK": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_phi0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_theta0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "Phi_r0": UniformDistribution([0.1, 2*pi*0.99], device=device),
    "T" : UniformDistribution([1, 6], device=device),
}


# class PopulationDistribution:
#     def __init__(self, distributions, data) -> None:
#         self.distributions = distributions
#         self.data = data

#     def draw_samples(self, x, size):
#         out = {}
#         for key in self.distributions.keys():
#             out[key] = self.distributions[key].draw_samples(**x[key], size=size)
#         return out


class PopulationDistribution:
    def __init__(self, distributions, data) -> None:
        self.distributions = distributions
        self.data = data

    def draw_samples(self, x, weight=1.0, size=500):
        out = {}

        self.weight = weight

        individual_pop_samples = []
        
        for key in self.distributions.keys():

            hyperparams = list(x[key].items())
            cleaned_hyperparams_A = {re.sub(r'_A$', '', k): v for k, v in hyperparams if k.endswith('_A')}
            cleaned_hyperparams_B = {re.sub(r'_B$', '', k): v for k, v in hyperparams if k.endswith('_B')}
            
            if not cleaned_hyperparams_A and not cleaned_hyperparams_B:  
                # If no '_A' or '_B' suffixes exist, use params directly
                out[key] = self.distributions[key].draw_samples(**x[key], size=int(size))
            
            else:

                choices = torch.bernoulli(torch.full((size,), self.weight)).bool()
                
                if cleaned_hyperparams_A == {'UNIFORM': {}}:
                    samples_A = self.distributions[key][0].draw_samples(size=size)
                    samples_B = self.distributions[key][1].draw_samples(**cleaned_hyperparams_B, size=size)

                elif cleaned_hyperparams_B == {'UNIFORM': {}}:
                    samples_A = self.distributions[key][0].draw_samples(**cleaned_hyperparams_A, size=size)
                    samples_B = self.distributions[key][1].draw_samples(size=size)

                else :
                    samples_A = self.distributions[key][0].draw_samples(**cleaned_hyperparams_A, size=size)
                    samples_B = self.distributions[key][1].draw_samples(**cleaned_hyperparams_B, size=size)

                # Select from A or B based on choices
                out[key] = torch.where(choices, samples_A, samples_B)

        return out


popdist_A = PopulationDistribution(distributions=distributions_A, data=None)

popdist_B = PopulationDistribution(distributions=distributions_B, data=None)

popdist_C = PopulationDistribution(distributions=distributions_C, data=None)

popdist_MIX = PopulationDistribution(distributions=POLULATION_MIX, data=None)

