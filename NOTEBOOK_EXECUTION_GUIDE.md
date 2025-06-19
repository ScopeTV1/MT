# Automated Notebook Execution Guide (Apple Silicon Optimized)

This guide provides instructions for executing all Jupyter notebooks in the thesis project sequentially while maximizing performance on Apple Silicon (M-series) chips.

## Apple Silicon Performance Optimizations

### Automatic Detection & Optimization
- **Hardware Detection**: Automatically detects Apple Silicon chips (M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M2 Ultra, M3, etc.)
- **Metal Performance Shaders (MPS)**: Enables GPU acceleration for PyTorch
- **Accelerate Framework**: Leverages Apple's optimized BLAS/LAPACK
- **ARM64 Threading**: Optimized threading for Apple's architecture

### Performance Features
- **Memory Optimization**: Uses 95% memory threshold for optimal performance
- **CPU Utilization**: Leverages 95% CPU threshold across all available cores
- **Optimized Environment**: Sets optimal environment variables for Apple Silicon
- **Metal Acceleration**: Enables TensorFlow Metal and PyTorch MPS when available

## Notebook Execution Order

The automated script executes these notebooks in sequence (non-archived notebooks only):
- `exploration_main_descriptive_statistics.ipynb`
- `exploration_02_OLS_main.ipynb`
- `exploration_03_RandomForest_main.ipynb`
- `exploration_03_RandomForest_SetB_main.ipynb`
- `exploration_04_XGB_main.ipynb`
- `exploration_04_XGB_SetB_main.ipynb`

## Usage

### Prerequisites

#### Option 1: Auto-Install Optimized Packages (Recommended for Apple Silicon)
```bash
# Run the Apple Silicon optimizer (automatically detects your hardware)
python install_apple_silicon_packages.py
```

#### Option 2: Manual Installation
```bash
# Install base requirements
pip install -r requirements.txt

# For Apple Silicon users - install optimized packages
pip install tensorflow-macos tensorflow-metal  # Apple's optimized TensorFlow
pip install torch torchvision torchaudio       # PyTorch with MPS support

# For Intel Macs - standard packages
pip install tensorflow torch torchvision torchaudio
```

### Running the Script
```bash
# Navigate to your project directory
cd "/Users/luis.m/Library/Mobile Documents/com~apple~CloudDocs/Documents ☁️/VSC Projects/Master_Thesis"

# Run the helper script
python execute_notebooks.py
```

### What Happens During Execution

1. **Resource Check**: Verifies system has enough free resources
2. **Backup Creation**: Creates `.ipynb.backup` files before execution
3. **Sequential Execution**: Runs one notebook at a time
4. **Output Preservation**: All cell outputs are saved in the original notebook
5. **Progress Logging**: Real-time updates in terminal and log files
6. **System Monitoring**: Continuous resource monitoring during execution

## Resource Management

### Apple Silicon Performance Mode
- **Memory Limit**: 95% (uses maximum available memory)
- **CPU Limit**: 95% (uses all available cores)
- **Threading**: Optimized for ARM64 architecture
- **GPU Acceleration**: Metal Performance Shaders when available
- **Minimal Monitoring**: Reduced overhead for maximum speed

### Optimizations Applied
- **Environment Variables**: 15+ optimized settings for Apple Silicon
- **Threading Libraries**: OMP, MKL, NumExpr, OpenBLAS, Accelerate
- **Memory Management**: Optimized malloc and multiprocessing
- **GPU Utilization**: TensorFlow Metal + PyTorch MPS support

## Output Files

- **Logs**: Saved in `logs/` directory with timestamps
- **Execution Report**: Detailed JSON report with timing and resource usage
- **Updated Notebooks**: Original notebooks with all outputs preserved

## Error Handling

- **Backup Restoration**: Failed notebooks are restored from backup
- **Graceful Failure**: Script continues with remaining notebooks if one fails
- **Resource Timeout**: Waits up to 10 minutes for resources to become available

## Monitoring Progress

The script provides real-time updates including:
- Current notebook being executed
- System resource usage (memory, CPU)
- Execution time for each notebook
- Success/failure status
- Final summary report

## Stopping Execution

Press `Ctrl+C` to gracefully stop execution. The script will:
- Complete the current notebook if possible
- Save all progress made so far
- Generate a partial execution report

## File Structure After Execution

```
notebooks/
├── exploration_02_OLS_main.ipynb          # ✅ With outputs
├── exploration_03_RandomForest_main.ipynb # ✅ With outputs
├── ... (other notebooks with outputs)
logs/
├── notebook_execution_20250618_143022.log
├── execution_report_20250618_143022.json
```

## Tips for Best Results

1. **Close Unnecessary Apps**: Free up system resources before running
2. **Plug in Power**: Ensure your Mac is plugged in for long executions
3. **Monitor Progress**: Check logs if you need to see detailed progress
4. **Backup Important Work**: The script creates backups, but external backups are always good

## Troubleshooting

- **"Jupyter not found"**: Install with `pip install jupyter`
- **Resource timeouts**: Close other applications to free up resources
- **Notebook errors**: Check individual notebook logs in the logs directory
- **Permissions issues**: Ensure you have write access to the notebooks directory

This helper script is designed to be "fire and forget" - start it and let it run while you're away!
