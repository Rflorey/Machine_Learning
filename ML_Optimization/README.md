# ML Optimization Examples

This repository contains Python scripts for machine learning optimization examples from the Machine Learning & AI Micro Boot Camp at Arizona State University (2023).

## Credit
These scripts were created as part of coursework for the Machine Learning & AI Micro Boot Camp – ARIZONA STATE UNIVERSITY (2023).

## Contents

- `hyperparameters_solution.py`: Demonstrates hyperparameter tuning using both GridSearchCV and RandomizedSearchCV with KNN classifiers.
- `make_blobs_data.py`: Creates synthetic imbalanced dataset using sklearn's make_blobs and saves it to CSV.

## Requirements

All required packages are listed in `requirements.txt`. To install dependencies:

```bash
pip install -r requirements.txt
```

## requirements.txt

```
matplotlib>=3.5.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## Usage

1. The scripts expect a directory structure with a data folder:
   ```
   ML_Optimization/
   ├── data/
   │   ├── numeric_bank.csv
   │   └── blobs_data.csv
   ```

2. Run the scripts:
   ```bash
   python make_blobs_data.py  # Creates synthetic dataset
   python hyperparameters_solution.py  # Runs hyperparameter optimization
   ```

## Features

- Hyperparameter optimization using:
  - Grid Search
  - Random Search
- Synthetic data generation with controlled imbalance
- Visualization of datasets
- Classification metrics reporting