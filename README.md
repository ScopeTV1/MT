# Predicting Operating Cash Flow: A Comparison of Machine Learning and Traditional Models

## Thesis Information

**Title:** Predicting Operating Cash Flow: A Comparison of Machine Learning and Traditional Models

**Author:** Luis Mauel

**Institution:** Maastricht University

**Program:** International Business

**Date:** 2025

## Repository Overview

This repository contains the complete implementation and analysis code for my master's thesis, which investigates the effectiveness of machine learning models compared to traditional forecasting methods for predicting operating cash flow. The research evaluates various machine learning algorithms (Random Forest, XGBoost) against conventional approaches (OLS regression) to determine which methodology provides more accurate predictions.

## Software Requirements

### Python Environment
- **Python Version**: 3.9 or higher (tested on Python 3.9-3.11)
- **Operating System**: Compatible with Windows, macOS, and Linux
- **Hardware**: Optimized for Apple Silicon (M1/M2 chips) but works on all architectures

### Required Python Packages

#### Core Data Science Libraries
- `pandas >= 1.5.0` - Data manipulation and analysis
- `numpy >= 1.24.0` - Numerical computing
- `scipy >= 1.10.0` - Scientific computing

#### Machine Learning Libraries
- `scikit-learn >= 1.2.0` - Traditional machine learning algorithms
- `xgboost >= 1.7.0` - Gradient boosting framework
- `statsmodels >= 0.14.0` - Statistical modeling and econometrics

#### Visualization Libraries
- `matplotlib >= 3.6.0` - Basic plotting
- `seaborn >= 0.12.0` - Statistical data visualization
- `plotly >= 5.14.0` - Interactive visualizations

#### Jupyter Notebook Environment
- `jupyter >= 1.0.0` - Jupyter notebook server
- `nbconvert >= 7.0.0` - Notebook conversion utilities
- `ipykernel >= 6.21.0` - IPython kernel for Jupyter

#### Additional Utilities
- `tqdm >= 4.64.0` - Progress bars
- `psutil >= 5.9.0` - System monitoring
- `openpyxl >= 3.1.0` - Excel file handling
- `xlsxwriter >= 3.0.0` - Excel file writing

## Installation Instructions

### 1. Clone the Repository
```bash
git clone [your-repository-url]
cd Master_Thesis
```

### 2. Set up Python Environment (Recommended)
```bash
# Create virtual environment
python -m venv thesis_env

# Activate virtual environment
# On Windows:
thesis_env\Scripts\activate
# On macOS/Linux:
source thesis_env/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

## Repository Structure

```
Master_Thesis/
├── src/                          # Core source code
│   ├── config.py                 # Configuration settings and parameters
│   └── processing.py             # Data processing and utility functions
├── notebooks/                    # Analysis notebooks (execution order)
│   ├── exploration_main_descriptive_statistics.ipynb  # [1] Descriptive statistics
│   ├── exploration_02_OLS_main.ipynb                  # [2] OLS regression analysis
│   ├── exploration_03_RandomForest_main.ipynb         # [3] Random Forest model
│   ├── exploration_03_RandomForest_SetB_main.ipynb    # [4] Random Forest variant
│   ├── exploration_04_XGB_main.ipynb                  # [5] XGBoost model
│   └── exploration_04_XGB_SetB_main.ipynb             # [6] XGBoost variant
├── data/                         # Data directory (raw data not included)
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed datasets
├── tables/                       # Generated tables and outputs
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Step-by-Step Reproduction Instructions

Execute notebooks in the following order to reproduce all thesis results:

#### Step 1: Descriptive Statistics
Open and run: `notebooks/exploration_main_descriptive_statistics.ipynb`
**Generates:** Descriptive statistics tables for thesis Chapter 4

#### Step 2: OLS Regression Analysis
Open and run: `notebooks/exploration_02_OLS_main.ipynb`
**Generates:** 
- OLS regression results table (`tables/ols_regression_table.tex`)
- Traditional forecasting performance metrics

#### Step 3: Random Forest Model
Open and run: `notebooks/exploration_03_RandomForest_main.ipynb`
**Generates:** Random Forest model performance metrics and feature importance plots

#### Step 4: Random Forest Model - Set B
Open and run: `notebooks/exploration_03_RandomForest_SetB_main.ipynb`
**Generates:** Alternative Random Forest model results for robustness testing

#### Step 5: XGBoost Model
Open and run: `notebooks/exploration_04_XGB_main.ipynb`
**Generates:** XGBoost model performance metrics and feature importance analysis

#### Step 6: XGBoost Model - Set B
Open and run: `notebooks/exploration_04_XGB_SetB_main.ipynb`
**Generates:** Alternative XGBoost model results for robustness testing

## Generated Outputs

After running all notebooks, the following tables and figures will be generated:

### Tables (LaTeX format for thesis)
- `tables/descriptive_statistics_stargazer.tex` - Descriptive statistics table
- `tables/ols_regression_table.tex` - OLS regression results table
- Model comparison tables (generated within notebooks)

### Figures
- Feature importance plots for Random Forest models
- Feature importance plots for XGBoost models
- Model performance comparison charts
- Residual analysis plots

## Key Research Findings

The analysis implements and compares:
- **Traditional Models**: Ordinary Least Squares (OLS) regression
- **Machine Learning Models**: Random Forest, XGBoost
- **Performance Metrics**: RMSE, MAE, R², and directional accuracy
- **Robustness Testing**: Multiple model configurations and variable sets

## Data Requirements

**Note:** Raw data is not included in this repository due to confidentiality. To reproduce results:
1. Place your dataset in `data/raw/` directory
2. Ensure data follows the expected format as described in the notebooks
3. Update file paths in `src/config.py` if necessary

## Troubleshooting

### Common Issues
1. **Missing data error**: Ensure your dataset is placed in `data/raw/` directory
2. **Package conflicts**: Use a fresh virtual environment
3. **Memory issues**: Close other applications when running large models

### Performance Optimization
- For large datasets: Consider increasing memory limits in notebook configurations
- For faster execution: Run notebooks individually and clear outputs between runs
- Monitor system resources during model training

## Academic Citation

If you use this code in your research, please cite:

```
Mauel, L. (2025). Predicting Operating Cash Flow: A Comparison of Machine Learning and Traditional Models. 
Master's Thesis, International Business Program, Maastricht University.
```

## License

This code is part of academic research conducted at Maastricht University. Please cite appropriately if using any part of this work.

---

*This repository represents the complete practical implementation of the thesis "Predicting Operating Cash Flow: A Comparison of Machine Learning and Traditional Models" submitted to Maastricht University as part of the International Business master's program.*
