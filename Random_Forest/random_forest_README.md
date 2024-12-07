# Random Forest Classification Examples

This repository contains various implementations of Random Forest classification algorithms, including balanced and imbalanced learning approaches. These scripts were created as part of the Machine Learning & AI Micro Boot Camp at Arizona State University (2023).

## Credit
These scripts are the result of coursework completed during the Machine Learning & AI Micro Boot Camp – Arizona State University (2023).

## Scripts Overview

- `random_forest_example.py`: Basic Random Forest implementation for malware detection
- `random_resampling_solution.py`: Random Forest with random undersampling for imbalanced data
- `synthetic_resampling_solution.py`: Implementation using synthetic resampling techniques
- `balanced_random_forest.py`: Comparison of standard Random Forest vs Balanced Random Forest

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Requirements

The scripts expect CSV files in the `Random_Forest/data/` directory:
- app_data.csv
- bank.csv
- sba_loans.csv

---

# requirements.txt

```
pandas
scikit-learn
imbalanced-learn
numpy
```
