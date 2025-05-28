"""
Data processing module for Master Thesis project.
Handles data loading, preparation, feature construction, and winsorization.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Union
from scipy.stats.mstats import winsorize
import src.config as config

# =============================================================================
# CONFIGURATION CONSTANTS (imported from config.py)
# =============================================================================

# Use constants from config file
ESSENTIAL_COLUMNS = config.ESSENTIAL_RAW_COLUMNS
FISCAL_YEAR_START = config.FISCAL_YEAR_START
FISCAL_YEAR_END = config.FISCAL_YEAR_END
FINANCIAL_SIC_RANGE = config.FINANCIAL_SIC_RANGE
UTILITY_SIC_RANGE = config.UTILITY_SIC_RANGE
RAW_ITEMS_TO_LAG = config.RAW_ITEMS_TO_LAG
SET_A_FEATURES = config.SET_A_FEATURES
CONTROL_DUMMIES = config.CONTROL_DUMMY_FEATURES
SET_B_FEATURES = config.SET_B_FEATURES
FINAL_COLUMNS_CORE = config.CORE_ID_COLUMNS + [config.DEPENDENT_VARIABLE]
ALL_FEATURE_COLUMNS = config.ALL_CONSTRUCTED_FEATURES

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def df_shape(df: pd.DataFrame) -> None:
    """Print DataFrame shape information."""
    rows, columns = df.shape
    print(f"The DataFrame has {rows} observations (rows) and {columns} columns.")


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load data from a CSV file with error handling."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to a CSV file with error handling."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """Validate that required columns exist in DataFrame and return available ones."""
    available_columns = [col for col in required_columns if col in df.columns]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    return available_columns


def _create_lags(df: pd.DataFrame, vars_to_lag: List[str], 
                id_col: str = 'gvkey', date_col: str = 'fyear', 
                lag_periods: int = 1) -> pd.DataFrame:
    """Create lagged variables grouped by entity and sorted by time."""
    print(f"  Creating lags for: {vars_to_lag}")
    df_copy = df.copy().sort_values(by=[id_col, date_col])
    
    for var in vars_to_lag:
        lag_col_name = f'{var}_lag{lag_periods}'
        if var in df_copy.columns:
            df_copy[lag_col_name] = df_copy.groupby(id_col)[var].shift(lag_periods)
        else:
            print(f"    Warning: Column '{var}' not found for lagging.")
            df_copy[lag_col_name] = np.nan
    
    return df_copy


def _initialize_columns(df: pd.DataFrame, column_names: List[str], 
                       default_value: Union[float, int] = np.nan) -> pd.DataFrame:
    """Initialize multiple columns with a default value."""
    df_copy = df.copy()
    for col in column_names:
        df_copy[col] = default_value
    return df_copy


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two series, handling zeros and NaNs."""
    return numerator / denominator.replace(0, np.nan)


# =============================================================================
# DATA PREPARATION FUNCTIONS
# =============================================================================


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data by selecting essential columns, filtering by fiscal year,
    and excluding financial and utility industries.
    """
    print(f"Original number of observations: {len(df)}")
    
    # Step 1: Select essential columns
    columns_to_keep = _validate_columns(df, ESSENTIAL_COLUMNS)
    df_selected = df[columns_to_keep].copy()
    print(f"Number of columns after selection: {len(df_selected.columns)}")
    
    # Step 2: Filter by fiscal year
    df_filtered_year = df_selected[
        (df_selected['fyear'] >= FISCAL_YEAR_START) & 
        (df_selected['fyear'] <= FISCAL_YEAR_END)
    ].copy()
    print(f"Observations after year filter ({FISCAL_YEAR_START}-{FISCAL_YEAR_END}): {len(df_filtered_year)}")
    
    # Step 3: Filter by industry (exclude financials and utilities)
    df_filtered_year['sic'] = pd.to_numeric(df_filtered_year['sic'], errors='coerce')
    
    is_financial = (
        (df_filtered_year['sic'] >= FINANCIAL_SIC_RANGE[0]) & 
        (df_filtered_year['sic'] <= FINANCIAL_SIC_RANGE[1])
    )
    is_utility = (
        (df_filtered_year['sic'] >= UTILITY_SIC_RANGE[0]) & 
        (df_filtered_year['sic'] <= UTILITY_SIC_RANGE[1])
    )
    
    df_final = df_filtered_year[~(is_financial | is_utility)].copy()
    print(f"Observations after excluding financial and utility firms: {len(df_final)}")
    
    return df_final


def drop_missing_final_vars_streamlined(df_with_features: pd.DataFrame, 
                                       list_of_columns_to_check: List[str]) -> pd.DataFrame:
    """
    Remove rows with missing values in specified final model variable columns.
    """
    initial_rows = len(df_with_features)
    print(f"\nStarting final dropna. Initial rows: {initial_rows}, DataFrame shape: {df_with_features.shape}")

    # Identify which columns are actually present
    actual_cols_for_dropna = _validate_columns(df_with_features, list_of_columns_to_check)
    
    if not actual_cols_for_dropna:
        print("  Warning: None of the specified columns for dropna check exist. No rows dropped.")
        return df_with_features.copy()

    df_analytical_sample = df_with_features.dropna(subset=actual_cols_for_dropna).copy()
    
    rows_dropped = initial_rows - len(df_analytical_sample)
    print(f"  Rows dropped due to NaNs in critical model variables: {rows_dropped}")
    print(f"  Shape of analytical sample AFTER final dropna: {df_analytical_sample.shape}")
    
    # Verification
    if not df_analytical_sample.empty:
        nan_count = df_analytical_sample[actual_cols_for_dropna].isnull().sum().sum()
        if nan_count == 0:
            print("  Verification: No NaNs in checked columns of analytical sample.")
        else:
            print("  Warning: NaNs still found in checked columns after dropna.")
            
    return df_analytical_sample


# =============================================================================
# FEATURE CONSTRUCTION FUNCTIONS
# =============================================================================

def _prepare_data_for_construction(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for feature construction with validity checks."""
    print("\nPerforming pre-calculation validity checks & preparations...")
    df_copy = df.copy()
    
    # Ensure 'at' is positive for scaling
    if 'at' in df_copy.columns:
        df_copy['at'] = pd.to_numeric(df_copy['at'], errors='coerce')
        df_copy.loc[df_copy['at'] <= 0, 'at'] = np.nan 
        if df_copy['at'].isnull().all():
            print("  Critical Warning: 'at' column is all NaN after positivity check.")
    else:
        print("  Critical Warning: 'at' column missing.")

    # Handle missing XRD (fill with 0 before scaling)
    if 'xrd' in df_copy.columns:
        df_copy['xrd'] = df_copy['xrd'].fillna(0)
        print("  Missing 'xrd' values filled with 0.")
    else:
        print("  Warning: 'xrd' column missing for XRD_Scaled_t.")

    # Prepare ipo_year for FirmAge
    if all(col in df_copy.columns for col in ['ipodate', 'fyear']):
        df_copy['ipodate_dt'] = pd.to_datetime(df_copy['ipodate'], errors='coerce')
        df_copy['ipo_year'] = df_copy['ipodate_dt'].dt.year
        print("  'ipo_year' created from 'ipodate'.")
    else:
        print("  Warning: 'ipodate' or 'fyear' missing. FirmAge_t cannot be calculated.")
        df_copy['ipo_year'] = np.nan
        df_copy['ipodate_dt'] = pd.to_datetime(pd.Series([None]*len(df_copy)))

    return df_copy


def _construct_dependent_variable(df: pd.DataFrame, id_col: str = 'gvkey', 
                                 date_col: str = 'fyear') -> pd.DataFrame:
    """Construct the dependent variable OCF_Scaled_t+1."""
    print("\nConstructing dependent variable...")
    df_copy = df.copy().sort_values(by=[id_col, date_col])
    
    # Initialize columns
    df_copy = _initialize_columns(df_copy, ['OCF_Scaled_t_plus_1', 'oancf_t_plus_1'])

    if all(col in df_copy.columns for col in ['oancf', 'at']):
        df_copy['oancf_t_plus_1'] = df_copy.groupby(id_col)['oancf'].shift(-1)
        df_copy['OCF_Scaled_t_plus_1'] = _safe_divide(df_copy['oancf_t_plus_1'], df_copy['at'])
        print("  OCF_Scaled_t_plus_1 created.")
    else:
        print("  Warning: Could not create OCF_Scaled_t_plus_1 due to missing 'oancf' or 'at'.")
    
    return df_copy


def _construct_set_a_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Construct Set A (OLS) predictors."""
    print("\nConstructing Set A (OLS) predictors...")
    df_copy = _initialize_columns(df, SET_A_FEATURES)

    # Feature construction with safe division
    if all(col in df_copy.columns for col in ['oancf', 'at_lag1']):
        df_copy['OCF_Scaled_Lag_t'] = _safe_divide(df_copy['oancf'], df_copy['at_lag1'])
    
    if all(col in df_copy.columns for col in ['ni', 'at']):
        df_copy['NI_Scaled_t'] = _safe_divide(df_copy['ni'], df_copy['at'])
    
    if all(col in df_copy.columns for col in ['ni', 'oancf', 'at']):
        df_copy['Accruals_Scaled_t'] = _safe_divide(df_copy['ni'] - df_copy['oancf'], df_copy['at'])
    
    if all(col in df_copy.columns for col in ['rect', 'rect_lag1', 'at']):
        df_copy['Delta_Rec_Scaled_t'] = _safe_divide(df_copy['rect'] - df_copy['rect_lag1'], df_copy['at'])
    
    if all(col in df_copy.columns for col in ['invt', 'invt_lag1', 'at']):
        df_copy['Delta_Inv_Scaled_t'] = _safe_divide(df_copy['invt'] - df_copy['invt_lag1'], df_copy['at'])
    
    if all(col in df_copy.columns for col in ['ap', 'ap_lag1', 'at']):
        df_copy['Delta_AP_Scaled_t'] = _safe_divide(df_copy['ap'] - df_copy['ap_lag1'], df_copy['at'])
    
    if all(col in df_copy.columns for col in ['dp', 'at']):
        df_copy['DP_Scaled_t'] = _safe_divide(df_copy['dp'], df_copy['at'])
    
    if 'at' in df_copy.columns:
        # Only calculate log for positive values
        valid_at = df_copy['at'] > 0
        df_copy.loc[valid_at, 'ln_at_t'] = np.log(df_copy.loc[valid_at, 'at'])

    print("  Set A predictors constructed.")
    return df_copy


def _construct_dummy_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Construct control dummy variables."""
    print("\nConstructing control dummy variables...")
    df_copy = df.copy()
    
    if 'fyear' in df_copy.columns:
        df_copy['ASC606_dummy'] = (df_copy['fyear'] >= 2018).astype(int)
        df_copy['ASC842_dummy'] = (df_copy['fyear'] >= 2019).astype(int)
        df_copy['TCJA_dummy'] = (df_copy['fyear'] >= 2018).astype(int)
        df_copy['COVID_dummy'] = ((df_copy['fyear'] == 2020) | (df_copy['fyear'] == 2021)).astype(int)
        print("  Dummy variables constructed.")
    else:
        print("  Warning: 'fyear' missing, dummy variables set to NaN.")
        df_copy = _initialize_columns(df_copy, CONTROL_DUMMIES)
    
    return df_copy

def _construct_set_b_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Construct Set B (additional ML) predictors."""
    print("\nConstructing Set B (additional ML) predictors...")
    df_copy = _initialize_columns(df, SET_B_FEATURES)

    # Feature construction mappings for cleaner code
    feature_mappings = [
        ('XSGA_Scaled_t', 'xsga', 'at'),
        ('XRD_Scaled_t', 'xrd', 'at'),
        ('CAPX_Scaled_t', 'capx', 'at'),
        ('CurrentRatio_t', 'act', 'lct'),
        ('DebtToAssets_t', 'lt', 'at'),
        ('OCFtoSales_t', 'oancf', 'sale'),
        ('InvTurnover_t', 'cogs', 'invt'),
        ('RecTurnover_t', 'sale', 'rect'),
        ('NI_Scaled_Lag_t', 'ni_lag1', 'at_lag1'),
        ('CapitalIntensity_t', 'ppent', 'at'),
        ('MkBk_t', 'mkvalt', 'ceq'),
    ]
    
    # Apply simple ratio calculations
    for feature_name, numerator_col, denominator_col in feature_mappings:
        if all(col in df_copy.columns for col in [numerator_col, denominator_col]):
            df_copy[feature_name] = _safe_divide(df_copy[numerator_col], df_copy[denominator_col])
    
    # Special cases requiring additional logic
    # Gross Profit Margin
    if 'sale' in df_copy.columns and (df_copy['sale'].fillna(0) > 0).any():
        if 'gp' in df_copy.columns:
            df_copy['GPM_t'] = _safe_divide(df_copy['gp'], df_copy['sale'])
        elif 'cogs' in df_copy.columns:
            df_copy['GPM_t'] = _safe_divide(df_copy['sale'] - df_copy['cogs'], df_copy['sale'])
    
    # Delta Sales Scaled
    if all(col in df_copy.columns for col in ['sale', 'sale_lag1', 'at']):
        df_copy['Delta_Sales_Scaled_t'] = _safe_divide(df_copy['sale'] - df_copy['sale_lag1'], df_copy['at'])
    
    # Firm Age
    if all(col in df_copy.columns for col in ['fyear', 'ipo_year']):
        df_copy['FirmAge_t'] = df_copy['fyear'] - df_copy['ipo_year']
        # Set invalid ages to NaN
        invalid_age_mask = (df_copy['FirmAge_t'] < 0) | df_copy['ipo_year'].isnull()
        df_copy.loc[invalid_age_mask, 'FirmAge_t'] = np.nan
    
    print("  Set B predictors constructed.")
    return df_copy


def _select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the final model variables and key identifiers."""
    print("\nSelecting final model variables and dropping intermediate columns...")
    
    # Define all conceptually final constructed variables
    final_columns_conceptual = FINAL_COLUMNS_CORE + ALL_FEATURE_COLUMNS
    
    # Select only columns that actually exist
    existing_final_columns = _validate_columns(df, final_columns_conceptual)
    
    df_model_ready = df[existing_final_columns].copy()
    print(f"  Shape of DataFrame after final column selection: {df_model_ready.shape}")
    print(f"  Final columns kept: {len(df_model_ready.columns)} columns")
    
    return df_model_ready


# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================

def create_all_model_features_orchestrated(df_input: pd.DataFrame, 
                                          start_construction_fyear: Optional[int] = None,
                                          id_col: str = 'gvkey', 
                                          date_col: str = 'fyear') -> pd.DataFrame:
    """
    Orchestrate the construction of all model features.
    
    Args:
        df_input: DataFrame containing raw Compustat items (initially screened)
        start_construction_fyear: Optional year to start construction from
        id_col: Entity identifier column
        date_col: Time period column
    
    Returns:
        DataFrame ready for modeling with all constructed features
    """
    df = df_input.copy()
    print(f"Starting feature construction. Initial df shape: {df.shape}")

    # Step 0: Initial sorting and optional year filter
    df = df.sort_values(by=[id_col, date_col])
    if start_construction_fyear:
        df = df[df[date_col] >= (start_construction_fyear - 1)].copy()
        print(f"  Filtered for construction period (from {start_construction_fyear-1}): {df.shape}")

    # Step 1: Create lagged raw variables
    df = _create_lags(df, RAW_ITEMS_TO_LAG, id_col=id_col, date_col=date_col)

    # Step 2: Pre-calculation data validity checks & preparations
    df = _prepare_data_for_construction(df)

    # Step 3: Construct dependent variable
    df = _construct_dependent_variable(df, id_col=id_col, date_col=date_col)
    
    # Step 4: Construct Set A (OLS) predictor variables
    df = _construct_set_a_predictors(df)

    # Step 5: Construct control variables (dummies)
    df = _construct_dummy_variables(df)

    # Step 6: Construct Set B (additional ML) predictor variables
    df = _construct_set_b_predictors(df)

    # Step 7: Select final columns & clean up
    df_model_ready = _select_final_columns(df)
    
    print(f"\nFeature construction complete. Final DataFrame shape: {df_model_ready.shape}")
    return df_model_ready


# =============================================================================
# WINSORIZATION FUNCTION
# =============================================================================

def annual_winsorize_variables(df_input: pd.DataFrame, 
                             columns_to_winsorize: List[str], 
                             year_column: str = 'fyear',
                             lower_limit: float = 0.01, 
                             upper_limit: float = 0.01) -> pd.DataFrame:
    """
    Perform annual winsorization on specified continuous columns.

    Args:
        df_input: Input DataFrame with year column and columns to winsorize
        columns_to_winsorize: List of column names to be winsorized
        year_column: Name of the fiscal year column
        lower_limit: Lower percentile limit for winsorizing (e.g., 0.01 for 1st percentile)
        upper_limit: Upper percentile limit for winsorizing (e.g., 0.01 for 99th percentile)

    Returns:
        DataFrame with specified columns winsorized annually
    """
    df = df_input.copy()
    print(f"\nStarting annual winsorization for {len(columns_to_winsorize)} columns...")
    
    if year_column not in df.columns:
        print(f"  Warning: Year column '{year_column}' not found. Returning original DataFrame.")
        return df

    # Validate columns to winsorize
    actual_cols_to_winsorize = _validate_columns(df, columns_to_winsorize)
    
    if not actual_cols_to_winsorize:
        print("  No valid columns found to winsorize. Returning original DataFrame.")
        return df

    # Winsorize each column
    for col in actual_cols_to_winsorize:
        print(f"  Winsorizing column: {col}")
        # Ensure column is numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Perform year-by-year winsorization
        try:
            df[col] = df.groupby(year_column)[col].transform(
                lambda x: winsorize(x.astype(float), limits=[lower_limit, upper_limit]) 
                          if x.notna().any() 
                          else x
            )
        except Exception as e:
            print(f"    Error winsorizing column {col}: {e}")

    print("Annual winsorization complete.")
    return df

# =============================================================================
# MAIN FUNCTION AND COMMAND-LINE INTERFACE
# =============================================================================

def process_data_pipeline(input_file_path: str, 
                         output_file_path: Optional[str] = None,
                         start_construction_fyear: Optional[int] = None,
                         perform_winsorization: bool = True,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Complete data processing pipeline following the workflow from exploration_02.ipynb.
    
    Args:
        input_file_path: Path to the raw CSV data file
        output_file_path: Optional path to save the processed data
        start_construction_fyear: Optional year to start feature construction from
        perform_winsorization: Whether to perform winsorization (default: True)
        verbose: Whether to print detailed progress information
    
    Returns:
        Processed DataFrame ready for modeling
    """
    if verbose:
        print("=" * 80)
        print("MASTER THESIS DATA PROCESSING PIPELINE")
        print("=" * 80)
    
    # Step 1: Load raw data
    if verbose:
        print("\n1. Loading raw data...")
    df_raw = load_data(input_file_path)
    if df_raw is None:
        raise ValueError("Failed to load data from the specified file path.")
    
    if verbose:
        df_shape(df_raw)
    
    # Step 2: Prepare data (filter columns, years, industries)
    if verbose:
        print("\n2. Preparing data (filtering columns, years, industries)...")
    df_prepared = prepare_data(df_raw)
    
    if verbose:
        df_shape(df_prepared)
    
    # Step 3: Create all model features
    if verbose:
        print("\n3. Creating all model features...")
    df_with_features = create_all_model_features_orchestrated(
        df_prepared, 
        start_construction_fyear=start_construction_fyear
    )
    
    if verbose:
        df_shape(df_with_features)
    
    # Step 4: Drop rows with missing critical variables
    if verbose:
        print("\n4. Removing rows with missing critical variables...")
    df_complete_cases = drop_missing_final_vars_streamlined(
        df_with_features, 
        config.FINAL_SET_A_AND_DEPENDENT
    )
    
    if verbose:
        df_shape(df_complete_cases)
    
    # Step 5: Winsorize continuous variables (optional)
    if perform_winsorization:
        if verbose:
            print("\n5. Winsorizing continuous variables...")
        df_final = annual_winsorize_variables(
            df_complete_cases, 
            config.COLUMNS_TO_WINSORIZE,
            lower_limit=config.WINSORIZATION_LOWER_LIMIT,
            upper_limit=config.WINSORIZATION_UPPER_LIMIT
        )
    else:
        if verbose:
            print("\n5. Skipping winsorization...")
        df_final = df_complete_cases.copy()
    
    if verbose:
        df_shape(df_final)
        print("\n" + "=" * 80)
        print("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final dataset contains {len(df_final)} observations")
        print(f"Final dataset contains {len(df_final.columns)} variables")
        print(f"Available features: {len([col for col in df_final.columns if col not in config.CORE_ID_COLUMNS])}")
    
    # Step 6: Save processed data (optional)
    if output_file_path:
        if verbose:
            print(f"\n6. Saving processed data to: {output_file_path}")
        save_data(df_final, output_file_path)
    
    return df_final

def split_data_chronologically(df, year_column, split_year):
    """Splits DataFrame chronologically."""
    if year_column not in df.columns:
        raise ValueError(f"Year column '{year_column}' not found.")
    
    train_df = df[df[year_column] <= split_year].copy()
    test_df = df[df[year_column] > split_year].copy()
    
    print(f"Training set: {len(train_df)} obs (Predictor years <= {split_year})")
    print(f"Test set: {len(test_df)} obs (Predictor years > {split_year})")
    return train_df, test_df



