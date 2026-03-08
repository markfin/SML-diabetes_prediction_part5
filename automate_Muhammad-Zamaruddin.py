
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df_input):
    """
    Performs a complete preprocessing pipeline on the input DataFrame.

    Steps include:
    1. Handling missing values (0s replaced with NaN, then imputed with median).
    2. Removing duplicate rows.
    3. Detecting and handling outliers (capping using IQR method).
    4. Standardizing numerical features using StandardScaler.

    Args:
        df_input (pd.DataFrame): The raw input DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for model training.
    """
    df = df_input.copy() # Work on a copy to avoid modifying the original DataFrame

    # 1. Handle Missing Values (0s as NaN, then median imputation)
    columns_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_with_zero_as_missing] = df[columns_with_zero_as_missing].replace(0, np.nan)
    for col in columns_with_zero_as_missing:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # 2. Handle Duplicate Data
    df.drop_duplicates(inplace=True)

    # 3. Detect and Handle Outliers (Capping)
    numerical_cols = df.columns.tolist()
    if 'Outcome' in numerical_cols: # Exclude target variable from outlier detection
        numerical_cols.remove('Outcome')

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

    # 4. Normalization or Standardization of Features
    # Exclude target variable from scaling
    features_to_scale = df.columns.drop('Outcome') if 'Outcome' in df.columns else df.columns

    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

if __name__ == '__main__':
    print("This script provides a 'preprocess_data' function for automated data preprocessing.")
    print("To use it, import the function into your Python environment. For example:")
    print("from automate_Muhammad_Zamaruddin import preprocess_data")
    print("raw_data = pd.read_csv('your_raw_data.csv')")
    print("preprocessed_data = preprocess_data(raw_data)")
    print("print(preprocessed_data.head())")
