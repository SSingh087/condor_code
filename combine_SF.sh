#!/bin/sh

POP=$1

echo "Combining SFs for $POP"
python combine_sfs.py --population $POP

echo "Data generation complete"