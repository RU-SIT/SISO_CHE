# iMAML Inner Step Analysis Tool

This tool provides comprehensive analysis of iMAML performance on wireless channel estimation tasks, specifically designed for UMi and TDL scenarios. It tracks and analyzes inner loop losses across different channels and training steps to provide detailed insights into the learning process.

## Features

- **Detailed Inner Loop Tracking**: Tracks losses at each step of the inner loop for every channel
- **Statistical Analysis**: Comprehensive statistical analysis of learning performance
- **Visualization**: Multiple types of plots and visualizations for analysis
- **Cross-Scenario Comparison**: Compare performance between UMi and TDL scenarios
- **Performance Metrics**: Calculate learning efficiency, consistency scores, and improvement metrics
- **Automated Reporting**: Generate detailed reports and summaries

## Files

- `imaml_inner_step_analyzer.py`: Main analysis tool with comprehensive functionality
- `run_imaml_analysis.py`: Simple runner script for easy execution
- `README_IMAML_ANALYSIS.md`: This documentation file

## Quick Start

### Basic Usage

```bash
# Analyze UMi scenario with default parameters
python run_imaml_analysis.py --scenario UMi

# Analyze both UMi and TDL scenarios
python run_imaml_analysis.py --scenario both

# Analyze with custom parameters
python run_imaml_analysis.py --scenario UMi --meta_steps 1000 --K_shot 5 --lam 3.0
```

### Advanced Usage

```bash
# Use the main analyzer directly with full control
python imaml_inner_step_analyzer.py \
    --umi_data_dir /path/to/umi/data \
    --tdl_data_dir /path/to/tdl/data \
    --save_dir ./analysis_results \
    --meta_steps 1000 \
    --N_way 2 \
    --K_shot 10 \
    --n_steps 2 \
    --lam 2.0
```

## Parameters

### Data Parameters
- `--umi_data_dir`: Path to UMi dataset directory
- `--tdl_data_dir`: Path to TDL dataset directory
- `--save_dir`: Directory to save analysis results

### Training Parameters
- `--meta_steps`: Number of meta training steps (default: 1000)
- `--N_way`: Number of ways for few-shot learning (default: 2)
- `--K_shot`: Number of shots (default: 10)
- `--inner_lr`: Inner loop learning rate (default: 1e-3)
- `--outer_lr`: Outer loop learning rate (default: 1e-4)
- `--n_steps`: Number of inner steps (default: 2)
- `--lam`: Regularization parameter (default: 2.0)

### System Parameters
- `--use_gpu`: Use GPU for training (default: True)
- `--scenarios`: Scenarios to analyze (default: ['UMi', 'TDL'])

## Output Files

The analysis generates the following files for each scenario:

### Data Files
- `imaml_tracking_data_{scenario}_{timestamp}.json`: Detailed tracking data in JSON format
- `imaml_tracking_data_{scenario}_{timestamp}.csv`: Tracking data in CSV format for easy analysis
- `imaml_channel_statistics_{scenario}_{timestamp}.json`: Channel-wise statistics
- `imaml_progression_analysis_{scenario}_{timestamp}.json`: Learning progression analysis

### Visualization Files
- `imaml_channel_box_plots_{scenario}.png`: Box plots for each channel
- `imaml_learning_progression_{scenario}.png`: Learning progression plots
- `imaml_improvement_distribution_{scenario}.png`: Improvement distribution plots
- `imaml_channel_performance_{scenario}.png`: Channel performance comparison
- `imaml_epoch_analysis_{scenario}.png`: Epoch-wise analysis plots

### Report Files
- `imaml_summary_report_{scenario}_{timestamp}.txt`: Comprehensive summary report

## Analysis Components

### 1. Inner Step Tracking
- Tracks initial and final losses for each channel
- Records improvement metrics and ratios
- Monitors learning consistency across epochs

### 2. Statistical Analysis
- Channel-wise performance statistics
- Learning efficiency calculations
- Consistency score analysis
- Improvement distribution analysis

### 3. Visualizations
- **Channel Box Plots**: Compare performance across different channels
- **Learning Progression**: Track learning curves over training epochs
- **Improvement Distribution**: Analyze improvement patterns
- **Channel Performance**: Compare channel-specific performance
- **Epoch Analysis**: Analyze trends over training epochs

### 4. Performance Metrics
- **Learning Efficiency**: Combines improvement ratio with consistency
- **Consistency Score**: Measures stability of learning across appearances
- **Improvement Ratio**: Relative improvement from initial to final loss
- **Mean Improvement**: Absolute improvement in loss values

## Interpretation Guide

### Good Learning Indicators ✓
- High improvement ratio (>0.1)
- High consistency score (>0.8)
- High learning efficiency (>0.05)
- Decreasing final losses over epochs
- Low variance in loss distributions

### Poor Learning Indicators ❌
- Low or negative improvement ratio
- Low consistency score (<0.5)
- High variance in improvements
- Increasing final losses over epochs
- Inconsistent learning across channel appearances

## Example Usage Scenarios

### 1. Quick Analysis
```bash
# Run quick analysis with default parameters
python run_imaml_analysis.py --scenario UMi --meta_steps 200
```

### 2. Comprehensive Analysis
```bash
# Run comprehensive analysis for both scenarios
python run_imaml_analysis.py --scenario both --meta_steps 2000 --K_shot 5
```

### 3. Parameter Sensitivity Analysis
```bash
# Test different lambda values
python run_imaml_analysis.py --scenario UMi --lam 1.0
python run_imaml_analysis.py --scenario UMi --lam 3.0
python run_imaml_analysis.py --scenario UMi --lam 5.0
```

### 4. Different Shot Configurations
```bash
# Test different K-shot values
python run_imaml_analysis.py --scenario UMi --K_shot 5
python run_imaml_analysis.py --scenario UMi --K_shot 10
python run_imaml_analysis.py --scenario UMi --K_shot 20
```

## Customization

### Adding New Scenarios
To add new scenarios, modify the data directory paths in the main analyzer:

```python
# In imaml_inner_step_analyzer.py
if scenario.upper() == 'NEW_SCENARIO':
    data_dir = "/path/to/new/scenario/data"
```

### Custom Metrics
Add new performance metrics by extending the `generate_channel_statistics` method:

```python
def generate_channel_statistics(self):
    # Add your custom metrics here
    stats_summary[channel_name]['custom_metric'] = calculate_custom_metric(data)
```

### Custom Visualizations
Add new visualization functions by extending the `create_visualizations` method:

```python
def create_visualizations(self):
    # Add your custom visualizations here
    self._create_custom_plot(df)
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**: Reduce batch size or use CPU mode
   ```bash
   python run_imaml_analysis.py --no_gpu --meta_steps 500
   ```

2. **Dataset Path Issues**: Verify data directory paths
   ```bash
   python run_imaml_analysis.py --umi_data_dir /correct/path/to/umi/data
   ```

3. **Long Training Times**: Reduce meta steps for quick analysis
   ```bash
   python run_imaml_analysis.py --meta_steps 200
   ```

### Performance Tips

- Use GPU for faster training (if available)
- Reduce meta_steps for quick analysis
- Use smaller K_shot values for faster training
- Monitor GPU memory usage during training

## Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm
- scipy

## License

This tool is part of the iMAML wireless communication project. Please refer to the main project license for usage terms.

## Support

For questions or issues, please refer to the main project documentation or create an issue in the project repository.
