""" import pandas as pd

# Load the CSV
df = pd.read_csv("C:/Users/49157/Documents/MATLAB/buck_converter_output.csv")

# Strip any extra spaces from column names just in case
df.columns = df.columns.str.strip()

# Filter rows where 'SwitchState' is zero
filtered_rows = df[df['SwitchState'] == 0]

# Print the filtered rows
print(filtered_rows) """

import pandas as pd

# Load the CSV
""" df = pd.read_csv("hybrid_sensor_data.csv") """
df = pd.read_csv("matlab_code/single_mode_sensor_data.csv")
df_subset = df.head(2000)
#df_subset.to_csv('sampled_data.csv', index=False)
df_subset.to_csv('sampled_data_1mode.csv', index=False)
print("First 2000 datapoints have been saved to sampled_data_1mode.csv")

# Get the total number of rows in the DataFrame
#total_rows = len(df)

# Decide how many rows to sample
#sample_size = min(2000, total_rows)

# Randomly sample rows (no replacement)
#sampled_df = df.sample(n=sample_size, random_state=42)

# Write to a new CSV
#sampled_df.to_csv('sampled_data.csv', index=False)

#print(f"Sampled {sample_size} rows (from {total_rows}) to 'sampled_data.csv'")

""" import pandas as pd

# Load the CSV
df = pd.read_csv("C:/Users/49157/Documents/MATLAB/buck_converter_output.csv")

# Get the number of rows
num_rows = len(df)

print(f"Number of rows: {num_rows}")
 """