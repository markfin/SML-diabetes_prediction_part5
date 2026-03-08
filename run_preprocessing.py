import pandas as pd
import numpy as np
import requests
import os
import sys

# Add the directory containing automate_Muhammad_Zamaruddin.py to sys.path
# This ensures the import works when run as a standalone script
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from automate_Muhammad_Zamaruddin import preprocess_data

# Define paths and filenames
base_dir = '/content/drive/MyDrive/Colab Notebooks/Demo9'
dataset_url = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
raw_dataset_filename = 'diabetes_prediction.csv'
preprocessed_dataset_filename = 'preprocessed_diabetes.csv'

raw_save_path = os.path.join(base_dir, raw_dataset_filename)
preprocessed_save_path = os.path.join(base_dir, preprocessed_dataset_filename)

# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)

# 1. Download the raw dataset
print(f"Downloading raw dataset from {dataset_url}...")
response = requests.get(dataset_url)
response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
with open(raw_save_path, 'wb') as f:
    f.write(response.content)
print(f"Raw dataset downloaded and saved successfully to '{raw_save_path}'.")

# 2. Load the raw dataset
print(f"Loading raw dataset from '{raw_save_path}'...")
df_raw = pd.read_csv(raw_save_path)
print("Raw DataFrame Head:")
print(df_raw.head())
print("Raw DataFrame Shape:", df_raw.shape)

# 3. Preprocess the data using the imported function
print("Starting data preprocessing...")
df_preprocessed = preprocess_data(df_raw.copy()) # Pass a copy to avoid modifying original if needed elsewhere
print("Data preprocessing completed.")

# 4. Save the preprocessed data
print(f"Saving preprocessed data to '{preprocessed_save_path}'...")
df_preprocessed.to_csv(preprocessed_save_path, index=False)
print(f"Preprocessed data saved successfully to '{preprocessed_save_path}'.")

print("
Preprocessing workflow completed successfully.")
