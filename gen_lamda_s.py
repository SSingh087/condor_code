#!/usr/bin/env python

import numpy as np
import argparse
import os 
import re

def parse_and_generate_data():
    parser = argparse.ArgumentParser(description="Generate data for training.")

    # Define known arguments
    parser.add_argument("--total_samples", type=int, required=True, help="Total number of samples to generate.")
    parser.add_argument("--file_name", type=str, required=True, help="Output file to save generated data.")

    # Parse known arguments first
    args, unknown = parser.parse_known_args()

    # Extract dynamic variables and their ranges
    dynamic_params = {}
    for i in range(0, len(unknown), 3):  # Arguments are in triplets: --key min max
        if unknown[i].startswith("--") and i + 2 < len(unknown):
            key = unknown[i][2:]  # Remove '--'
            try:
                lower_bound = float(unknown[i + 1])
                upper_bound = float(unknown[i + 2])
                dynamic_params[key] = (lower_bound, upper_bound)
            except ValueError:
                raise ValueError(f"Invalid bounds for {unknown[i]}: {unknown[i + 1]}, {unknown[i + 2]}")

    # Generate data
    N = args.total_samples
    data = []
    for key, (param_min, param_max) in dynamic_params.items():
        data.append(np.random.uniform(param_min, param_max, N))
        print(key, param_min, param_max)

    data = np.asarray(data).T  # Transpose for proper formatting

    # Create directory if it doesn't exist
    directory = os.path.dirname(args.file_name)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Save data to the file
    np.save(args.file_name, data)

    print(f"Generated parameters: {list(dynamic_params.keys())}")
    print(f"Data shape: {data.shape}")
    print(f"Data saved to {args.file_name}")

if __name__ == "__main__":
    parse_and_generate_data()
