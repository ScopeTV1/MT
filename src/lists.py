# lists.py

# --- Lists of Raw Compustat Items ---

raw_items_for_identification = [
    'gvkey', 'datadate', 'fyear', 'conm', 'tic',
]

raw_items_for_screening = [
    'sic', 'pddur', 'curncd', 'ipodate',
    # 'consol', 'indfmt', 'datafmt', 'popsrc' # Add if filtering these in Python
]

raw_items_for_set_A_predictors_components = [
    'oancf', 'at', 'ni', 'rect', 'invt', 'ap', 'dp'
]

# Names for the dummy control variables (created from fyear)
control_dummy_variable_names = [
    'ASC606_dummy', 'ASC842_dummy', 'TCJA_dummy', 'COVID_dummy'
]

# Raw Compustat items needed *additionally* for Set B variables
raw_items_for_set_B_truly_additional_components = [
    'xsga', 'xrd', 'capx', 'act', 'lct', 'lt', 'sale', 'cogs',
    'gp',
    'ppent', 'mkvalt', 'ceq'
]

vars_to_lag = [ 
    'at',      # Needed for at_{t-1} (for scaling OCF_Scaled_Lag_t and NI_Scaled_Lag_t)
    'ni',      # Needed for ni_{t-1} (for NI_Scaled_Lag_t)
    'rect',    # Needed for rect_{t-1} (for ΔRec_Scaled_t)
    'invt',    # Needed for invt_{t-1} (for ΔInv_Scaled_t)
    'ap',      # Needed for ap_{t-1} (for ΔAP_Scaled_t)
    'sale'
]