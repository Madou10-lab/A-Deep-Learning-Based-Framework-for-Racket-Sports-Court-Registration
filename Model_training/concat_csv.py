import pandas as pd

# Step 1: Read CSVs into DataFrames
df1 = pd.read_csv('Data/Labels/metadata_dataset_labels.csv')
df2 = pd.read_csv('Data/Labels/Test_set/metadata_dataset_labels.csv')

# Step 2: Concatenate DataFrames
concatenated_df = pd.concat([df1, df2], ignore_index=True)

# Step 3: Save to new CSV
concatenated_df.to_csv('final_file.csv', index=False)
