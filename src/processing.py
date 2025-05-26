import pandas as pd
import numpy as np
import src.lists as lists

def df_shape(df):
    # Get the number of rows and columns
    rows, columns = df.shape
    print(f"The DataFrame has {rows} observations (rows) and {columns} columns.")


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_data(df, file_path):
    """
    Save DataFrame to a CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def prepare_data(df):
    """
    Prepare the data by selecting essential columns, filtering by fiscal year,
    and excluding financial and utility industries.
    """
    # --- Step 1: Define Essential Columns to Keep ---
    essential_columns = [
        # Identifiers & Date
        'gvkey', 'datadate', 'fyear', 'conm', 'tic', 'cusip', 'cik',
        # Industry Code
        'sic',
        # Data Quality/Screening Variables
        'curncd', 'pddur',
        # Variables for Dependent Variable (OCF_Scaled)
        'oancf', 'at',
        # Variables for OLS Predictors (Set A)
        'ni', 'rect', 'invt', 'ap', 'dp',
        # Variables for Additional ML Predictors (Set B)
        'xsga', 'xrd', 'capx', 'act', 'lct', 'lt', 'sale', 'gp', 'cogs',
        'ppent', 'mkvalt', 'ceq', 'ipodate'
    ]

    # Keep only essential columns
    columns_to_keep = [col for col in essential_columns if col in df.columns]
    df_selected_cols = df[columns_to_keep].copy()

    print(f"Original number of observations: {len(df)}")
    print(f"Number of columns after selection: {len(df_selected_cols.columns)}")

    # --- Step 2: Filter by Fiscal Year ---
    start_year = 2000
    end_year = 2023
    df_filtered_year = df_selected_cols[(df_selected_cols['fyear'] >= start_year) & (df_selected_cols['fyear'] <= end_year)].copy()
    print(f"Observations after year filter ({start_year}-{end_year}): {len(df_filtered_year)}")

    # --- Step 3: Filter by Industry (Exclude Financials and Utilities) ---
    df_filtered_year['sic'] = pd.to_numeric(df_filtered_year['sic'], errors='coerce')

    financial_sic_min = 6000
    financial_sic_max = 6999
    utility_sic_min = 4900
    utility_sic_max = 4999

    is_financial = (df_filtered_year['sic'] >= financial_sic_min) & (df_filtered_year['sic'] <= financial_sic_max)
    is_utility = (df_filtered_year['sic'] >= utility_sic_min) & (df_filtered_year['sic'] <= utility_sic_max)

    df_filtered_industry = df_filtered_year[~(is_financial | is_utility)].copy()
    print(f"Observations after excluding financial and utility firms: {len(df_filtered_industry)}")

    df_final_filters = df_filtered_industry.copy()

    return df_final_filters


def missing_values(df, drop_set_b=False):
    """
    Remove rows with missing values in essential predictor variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to process
    drop_set_b : bool, default=True
        Whether to drop rows with missing values in Set B predictor components
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with rows containing missing values removed
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Always drop rows with missing values in Set A predictors
    df_clean = df_clean.dropna(subset=lists.raw_items_for_set_A_predictors_components)
    
    # Optionally drop rows with missing values in Set B predictors
    if drop_set_b:
        df_clean = df_clean.dropna(subset=lists.raw_items_for_set_B_truly_additional_components)
    
    return df_clean

def create_lagged_variables(df, lag_years=1, vars_to_lag=None):
    """
    Create lagged variables for specified variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to process
    lag_years : int, default=1
        The number of years to lag the variables
    vars_to_lag : list, default=None
        List of variables to create lags for
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with lagged variables added
    """
    df_screened = df.copy()
    
    if vars_to_lag is None:
        print("No variables specified for lagging. Using default Set A predictors.")
        
    for var in vars_to_lag:
        df_screened[f'{var}_lag1'] = df_screened.groupby('gvkey')[var].shift(1)
    
    return df_screened