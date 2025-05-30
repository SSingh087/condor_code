#!/bin/sh

# =====================
#  POPULATION TRAINING
# =====================

NUM_NEURONS=$1
LAYERS=$2
POPULATION=$3

python train_model.py \
    --x_data_loc /data/wiay/postgrads/shashwat/EMRI_data/SF_MODEL_TEST/hyperparams_$POPULATION.npy \
    --y_data_loc /data/wiay/postgrads/shashwat/EMRI_data/SF_MODEL_TEST/sf_$POPULATION.npy \
    --train_cat SF \
    --num_neurons $NUM_NEURONS \
    --layers $LAYERS \
    --train_test_frac 0.8 \
    --learning_rate .001 \
    --n_epochs 10000 \
    --n_batches 1024 \
    --update_every 1000 \
    --verbose True \
    --outdir MODEL_hyperparam_SF_$POPULATION \
    --device cuda

echo "Training SF for $POPULATION with $NUM_NEURONS x $LAYERS complete"