#!/bin/sh

# ===========================
# A POPULATION
# ===========================

# mass dist of the MBH
A_PRIOR_XC_M_LOW=6.5
A_PRIOR_XC_M_HIGH=9.5

# mass dist of the CO
A_PRIOR_ALPHA_MU_LOW=-1.0
A_PRIOR_ALPHA_MU_HIGH=1.0

# spin of MBH
A_PRIOR_ALPHA_A_LOW=10
A_PRIOR_ALPHA_A_HIGH=15

A_PRIOR_BETA_A_LOW=5
A_PRIOR_BETA_A_HIGH=10

# eccentricity
A_PRIOR_ALPHA_E0_LOW=5
A_PRIOR_ALPHA_E0_HIGH=10

A_PRIOR_BETA_E0_LOW=1 
A_PRIOR_BETA_E0_HIGH=5

# ===========================
# B POPULATION
# this population is derived from https://doi.org/10.1093/mnras/stad1397
# ===========================

B_PRIOR_LAMBDA_M_LOW=-3
B_PRIOR_LAMBDA_M_HIGH=-1.01

# mass dist of the CO
B_PRIOR_LAMBDA_MU_LOW=-4
B_PRIOR_LAMBDA_MU_HIGH=-1.5

# spin 
B_PRIOR_MU_A_LOW=0.1
B_PRIOR_MU_A_HIGH=0.7

B_PRIOR_SIGMA_A_LOW=0.001
B_PRIOR_SIGMA_A_HIGH=0.05


# ===========================
# C POPULATION
# ===========================
# mass dist of the MBH
C_PRIOR_MU_M_LOW=6.5
C_PRIOR_MU_M_HIGH=9.5

# =======
# WEIGHTS
# =======
PRIOR_WEIGHT_LOW=0.001
PRIOR_WEIGHT_HIGH=0.99


TOTAL_SAMPLES=100000

ID=$1

POP=$2

if [ "$POP" = "A" ]; then
    echo "Generating hyperparameter dataset for population A"
    python gen_lamda_s.py \
        --total_samples $TOTAL_SAMPLES \
        --xc_M $A_PRIOR_XC_M_LOW $A_PRIOR_XC_M_HIGH \
        --alpha_mu $A_PRIOR_ALPHA_MU_LOW $A_PRIOR_ALPHA_MU_HIGH \
        --alpha_a $A_PRIOR_ALPHA_A_LOW $A_PRIOR_ALPHA_A_HIGH --beta_a $A_PRIOR_BETA_A_LOW $A_PRIOR_BETA_A_HIGH \
        --alpha_e0 $A_PRIOR_ALPHA_E0_LOW $A_PRIOR_ALPHA_E0_HIGH --beta_e0 $A_PRIOR_BETA_E0_LOW $A_PRIOR_BETA_E0_HIGH \
        --file_name /data/wiay/postgrads/shashwat/EMRI_data/SF_DATA_MODEL/hyperparams_$POP.npy


# ===========================
# B POPULATION
# this population is derived from https://doi.org/10.1093/mnras/stad1397
# ===========================
elif [ "$POP" = "B" ]; then
    echo "Generating hyperparameter dataset for population B"
    python gen_lamda_s.py \
        --total_samples $TOTAL_SAMPLES \
        --lambda_M $B_PRIOR_LAMBDA_M_LOW $B_PRIOR_LAMBDA_M_HIGH \
        --lambda_mu $B_PRIOR_LAMBDA_MU_LOW $B_PRIOR_LAMBDA_MU_HIGH \
        --mu_a $B_PRIOR_MU_A_LOW $B_PRIOR_MU_A_HIGH --sigma_a $B_PRIOR_SIGMA_A_LOW $B_PRIOR_SIGMA_A_HIGH \
        --file_name /data/wiay/postgrads/shashwat/EMRI_data/SF_DATA_MODEL/hyperparams_$POP.npy


elif [ "$POP" = "C" ]; then
    echo "Generating hyperparameter dataset for population C"
    python gen_lamda_s.py \
        --total_samples $TOTAL_SAMPLES \
        --mu_M $C_PRIOR_MU_M_LOW $C_PRIOR_MU_M_HIGH \
        --alpha_mu $A_PRIOR_ALPHA_MU_LOW $A_PRIOR_ALPHA_MU_HIGH \
        --alpha_a $A_PRIOR_ALPHA_A_LOW $A_PRIOR_ALPHA_A_HIGH --beta_a $A_PRIOR_BETA_A_LOW $A_PRIOR_BETA_A_HIGH \
        --alpha_e0 $A_PRIOR_ALPHA_E0_LOW $A_PRIOR_ALPHA_E0_HIGH --beta_e0 $A_PRIOR_BETA_E0_LOW $A_PRIOR_BETA_E0_HIGH \
        --file_name /data/wiay/postgrads/shashwat/EMRI_data/SF_DATA_MODEL/hyperparams_$POP.npy

elif [ "$POP" = "MIX" ]; then
    echo "Generating hyperparameter dataset for population A+B"
    python gen_lamda_s.py \
        --total_samples $TOTAL_SAMPLES \
        --A_lambda_M $A_PRIOR_LAMBDA_M_LOW $A_PRIOR_LAMBDA_M_HIGH \
        --A_alpha_mu $A_PRIOR_ALPHA_MU_LOW $A_PRIOR_ALPHA_MU_HIGH \
        --A_alpha_a $A_PRIOR_ALPHA_A_LOW $A_PRIOR_ALPHA_A_HIGH --A_beta_a $A_PRIOR_BETA_A_LOW $A_PRIOR_BETA_A_HIGH \
        --A_alpha_e0 $A_PRIOR_ALPHA_E0_LOW $A_PRIOR_ALPHA_E0_HIGH --A_beta_e0 $A_PRIOR_BETA_E0_LOW $A_PRIOR_BETA_E0_HIGH \
        --B_lambda_M $B_PRIOR_LAMBDA_M_LOW $B_PRIOR_LAMBDA_M_HIGH \
        --B_lambda_mu $B_PRIOR_LAMBDA_MU_LOW $B_PRIOR_LAMBDA_MU_HIGH \
        --B_mu_a $B_PRIOR_MU_A_LOW $B_PRIOR_MU_A_HIGH --B_sigma_a $B_PRIOR_SIGMA_A_LOW $B_PRIOR_SIGMA_A_HIGH \
        --WEIGHT $PRIOR_WEIGHT_LOW $PRIOR_WEIGHT_HIGH\
        --file_name /data/wiay/postgrads/shashwat/EMRI_data/SF_DATA_MODEL/hyperparams_$POP.npy

fi

echo "Generating SF for population $POP"
python gen_SF.py --population $POP --sample_index $ID --work_dir /data/wiay/postgrads/shashwat/EMRI_data/SF_DATA_MODEL/