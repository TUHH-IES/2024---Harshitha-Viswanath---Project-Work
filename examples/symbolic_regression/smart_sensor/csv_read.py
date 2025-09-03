"""
csv_read.py

This script reads the MATLAB-generated CSV file, selects the first 2000 rows,
and prepares a sampled dataset for segmentation and group identification.

The sampled dataset is saved as 'sampled_data.csv' in the current directory.

Usage:
    python csv_read.py 

Example:
    run python csv_read.py by replacing the path to matlab output file here 
    df = pd.read_csv("matlab_code/hybrid_sensor_data.csv")

Inputs:
    - input_file (str): Path to the MATLAB CSV output file.

Outputs:
    - sampled_data.csv: Contains the first 2000 rows from the input file.

Dependencies:
    - pandas

Assumptions:
    - The input CSV file must exist and contain at least 2000 rows.
    - The CSV file is expected to be MATLAB-formatted with standard delimiters.

"""

import pandas as pd

# Load the CSV
df = pd.read_csv("matlab_code/hybrid_sensor_data.csv")
#df = pd.read_csv("single_mode_sensor_data.csv")
df_subset = df.head(2000)
df_subset.to_csv('sampled_data.csv', index=False)
#df_subset.to_csv('sampled_data_1mode.csv', index=False)
print("First 2000 datapoints have been saved to sampled_data_1mode.csv")

