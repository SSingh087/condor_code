#!/usr/bin/env python

import numpy as np
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--population", type=str, required=True)
args = parser.parse_args()

def combine_npys(input_pattern, output_file):
    """
    Load multiple .npy files matching input_pattern,
    combine them into one numpy array, and save to output_file.
    """
    file_list = sorted(glob.glob(input_pattern))
    if not file_list:
        raise ValueError(f"No files found matching pattern: {input_pattern}")
    
    data_list = []
    for file_path in file_list:
        print(f"Loading {file_path}")
        data = np.load(file_path, allow_pickle=True)
        data_list.append(data)

    print(f"Saving combined data to {output_file}")
    np.save(output_file, np.array(data_list))

if __name__ == "__main__":
    input_pattern = f"/data/wiay/postgrads/shashwat/EMRI_data/SF_DATA_MODEL/sf_{args.population}/sf_{args.population}_*.npy"
    output_file = f"/data/wiay/postgrads/shashwat/EMRI_data/SF_DATA_MODEL/sf_{args.population}.npy"
    
    combine_npys(input_pattern, output_file)
