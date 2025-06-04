"""
Configuration file for Master Thesis project.
Contains all constants, column lists, and configuration parameters.
"""

# =============================================================================
# RAW DATA COLUMN DEFINITIONS
# =============================================================================

# Identification columns
RAW_IDENTIFICATION_COLUMNS = [
    'gvkey', 'datadate', 'fyear', 'conm', 'tic', 'cusip', 'cik'
]

# Screening and filtering columns
RAW_SCREENING_COLUMNS = [
    'sic', 'pddur', 'curncd', 'ipodate',
    # 'consol', 'indfmt', 'datafmt', 'popsrc' # Add if filtering these in Python
]

# Raw Compustat items for Set A (OLS) predictor components
RAW_SET_A_COMPONENTS = [
    'oancf', 'at', 'ni', 'rect', 'invt', 'ap', 'dp'
]

# Raw Compustat items needed additionally for Set B (ML) predictor components
RAW_SET_B_ADDITIONAL_COMPONENTS = [
    'xsga', 'xrd', 'capx', 'act', 'lct', 'lt', 'sale', 'cogs',
    'gp', 'ppent', 'mkvalt', 'ceq'
]

# Variables that need lagged versions (t-1)
RAW_ITEMS_TO_LAG = [ 
    'at',      # Needed for at_{t-1} (for scaling OCF_Scaled_Lag_t and NI_Scaled_Lag_t)
    'ni',      # Needed for ni_{t-1} (for NI_Scaled_Lag_t)
    'rect',    # Needed for rect_{t-1} (for ΔRec_Scaled_t)
    'invt',    # Needed for invt_{t-1} (for ΔInv_Scaled_t)
    'ap',      # Needed for ap_{t-1} (for ΔAP_Scaled_t)
    'sale'     # Needed for sale_{t-1} (for Δ Sales_Scaled_t)
]

# All essential raw columns (combination of above)
ESSENTIAL_RAW_COLUMNS = (
    RAW_IDENTIFICATION_COLUMNS + 
    RAW_SCREENING_COLUMNS + 
    RAW_SET_A_COMPONENTS + 
    RAW_SET_B_ADDITIONAL_COMPONENTS
)

# =============================================================================
# CONSTRUCTED FEATURE DEFINITIONS
# =============================================================================

# Set A (OLS) predictor feature names
SET_A_FEATURES = [
    'OCF_Scaled_Lag_t', 'NI_Scaled_t', 'Accruals_Scaled_t',
    'Delta_Rec_Scaled_t', 'Delta_Inv_Scaled_t', 'Delta_AP_Scaled_t',
    'DP_Scaled_t', 'ln_at_t'
]

# Control dummy variable names (created from fyear)
CONTROL_DUMMY_FEATURES = [
    'ASC606_dummy', 'ASC842_dummy', 'TCJA_dummy', 'COVID_dummy', 'ASC606_TCJA_combined_dummy'
]

# Clean dummy variables for OLS (resolves ASC606/TCJA multicollinearity)
OLS_DUMMY_FEATURES = [
    'ASC606_TCJA_combined_dummy', 'ASC842_dummy', 'COVID_dummy'
]

# Final dummy variables for OLS thesis table (only variables with variance in training set)
OLS_DUMMY_FEATURES_FINAL = [
    'ASC606_TCJA_combined_dummy'
]

# Set B (Additional ML) predictor feature names
SET_B_FEATURES = [
    'XSGA_Scaled_t', 'XRD_Scaled_t', 'CAPX_Scaled_t', 'CurrentRatio_t',
    'DebtToAssets_t', 'OCFtoSales_t', 'InvTurnover_t', 'RecTurnover_t',
    'GPM_t', 'Delta_Sales_Scaled_t', 'NI_Scaled_Lag_t',
    'CapitalIntensity_t', 'MkBk_t', 'FirmAge_t'
]

# Dependent variable name
DEPENDENT_VARIABLE = 'OCF_Scaled_t_plus_1'

# =============================================================================
# FINAL MODEL VARIABLE LISTS
# =============================================================================

# Set A predictors plus dependent variable (for missing data filtering)
FINAL_SET_A_AND_DEPENDENT = SET_A_FEATURES + [DEPENDENT_VARIABLE]

# All constructed features (for final column selection)
ALL_CONSTRUCTED_FEATURES = SET_A_FEATURES + CONTROL_DUMMY_FEATURES + SET_B_FEATURES

# Core identification columns to keep in final dataset
CORE_ID_COLUMNS = ['gvkey', 'fyear', 'datadate']

# =============================================================================
# WINSORIZATION CONFIGURATION
# =============================================================================

# Continuous variables for winsorization (excludes dummies and discrete variables)
CONTINUOUS_FEATURES_FOR_WINSORIZATION = [
    # Dependent variable
    DEPENDENT_VARIABLE,
    # Set A continuous predictors
    'OCF_Scaled_Lag_t', 'NI_Scaled_t', 'Accruals_Scaled_t',
    'Delta_Rec_Scaled_t', 'Delta_Inv_Scaled_t', 'Delta_AP_Scaled_t',
    'DP_Scaled_t', 'ln_at_t',
    # Set B continuous predictors (excluding FirmAge_t which is typically discrete)
    'XSGA_Scaled_t', 'XRD_Scaled_t', 'CAPX_Scaled_t', 'CurrentRatio_t',
    'DebtToAssets_t', 'OCFtoSales_t', 'InvTurnover_t', 'RecTurnover_t',
    'GPM_t', 'Delta_Sales_Scaled_t', 'NI_Scaled_Lag_t',
    'CapitalIntensity_t', 'MkBk_t'
]

# Ensure uniqueness and sort for consistency
COLUMNS_TO_WINSORIZE = sorted(list(set(CONTINUOUS_FEATURES_FOR_WINSORIZATION)))

# =============================================================================
# DATA FILTERING CONFIGURATION
# =============================================================================

# Time period configuration
FISCAL_YEAR_START = 2000
FISCAL_YEAR_END = 2023

# Industry exclusion criteria (SIC codes)
FINANCIAL_SIC_RANGE = (6000, 6999)
UTILITY_SIC_RANGE = (4900, 4999)

# Winsorization parameters
WINSORIZATION_LOWER_LIMIT = 0.01  # 1st percentile
WINSORIZATION_UPPER_LIMIT = 0.01  # 99th percentile

# =============================================================================
# BACKWARD COMPATIBILITY (for existing code)
# =============================================================================

# Legacy names for backward compatibility
raw_items_for_identification = RAW_IDENTIFICATION_COLUMNS
raw_items_for_screening = RAW_SCREENING_COLUMNS
raw_items_for_set_A_predictors_components = RAW_SET_A_COMPONENTS
raw_items_for_set_B_truly_additional_components = RAW_SET_B_ADDITIONAL_COMPONENTS
vars_to_lag = RAW_ITEMS_TO_LAG
control_dummy_variable_names = CONTROL_DUMMY_FEATURES
final_set_A_predictor_names_and_dependent = FINAL_SET_A_AND_DEPENDENT
final_set_A_continuous_predictors = SET_A_FEATURES
final_set_B_continuous_predictors = SET_B_FEATURES
dependent_variable_name = DEPENDENT_VARIABLE
columns_to_winsorize = COLUMNS_TO_WINSORIZE
