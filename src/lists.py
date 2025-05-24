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

