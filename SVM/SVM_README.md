# SVM Classification Examples

This repository contains example scripts demonstrating Support Vector Machine (SVM) classification using scikit-learn. These scripts were created as part of coursework from the Machine Learning & AI Micro Boot Camp at Arizona State University (2023).

## Credit
These scripts are the result of coursework completed during the Machine Learning & AI Micro Boot Camp – Arizona State University (2023).

## Scripts

### complex_data_example.py
Demonstrates SVM classification on a spiral dataset, showing the limitations of SVM with complex non-linear patterns. The script includes data visualization and model performance evaluation.

### SVM_predict_example.py
Implements SVM classification for room occupancy prediction using environmental sensor data (temperature, humidity, light, and CO2 levels). The script showcases data preprocessing, model training, and accuracy evaluation.

## Dataset Sources
- spirals.csv: Synthetic dataset for demonstrating non-linear classification
- occupancy.csv: Room occupancy detection dataset from:
  - Source: "Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models." Luis M. Candanedo, Véronique Feldheim. Energy and Buildings. Volume 112, 15 January 2016, Pages 28-39.

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the scripts using Python:
```bash
python complex_data_example.py
python SVM_predict_example.py
```