# Machine Learning Analysis Tools

A comprehensive collection of machine learning scripts and tools, specializing in clustering analysis and data preprocessing techniques.

## Overview

This repository contains robust implementations of various machine learning algorithms and analysis tools, with a primary focus on clustering analysis. The tools are designed to be both educational for learning machine learning concepts and practical for real-world data analysis applications.

## Features

- Automated clustering analysis with built-in visualization
- Data preprocessing utilities including scaling and encoding
- Optimal cluster number determination using the elbow method
- Customer segmentation capabilities
- Standardized data scaling
- Support for both CSV files and DataFrame inputs

## Scripts

### Core Components

- `simple_clustering_class.py`: A comprehensive class for clustering analysis that includes:
  - Data loading and preprocessing
  - Automated feature scaling
  - Elbow curve analysis
  - Cluster visualization
  - Support for dummy variable creation
  - Standardized scaling options

- `K_means_example.py`: Customer segmentation implementation that demonstrates:
  - Card type encoding
  - Income scaling
  - Multiple cluster number comparison (k=4 and k=5)
  - Visualization of customer segments

- `best_K_means_example.py`: Optimal K-value determination script featuring:
  - Elbow method implementation
  - Inertia score calculation
  - Automated cluster analysis
  - Visual elbow curve plotting

- `standard_scaler_example.py`: Data preprocessing example showing:
  - Standard scaling implementation
  - Categorical variable encoding
  - DataFrame transformation techniques

## Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed and the following dependencies:
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/machine-learning-analysis-tools.git
cd machine-learning-analysis-tools
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Simple Clustering Analysis
```python
from simple_clustering_class import SimpleClusterAnalysis

# Initialize and load data
clustering = SimpleClusterAnalysis()
clustering.load_data("your_data.csv")

# Preprocess data
clustering.create_dummies(['categorical_column'])
clustering.preprocess(
    drop_columns=["unnecessary_column"],
    scale_columns={"numeric_column": 1000}
)

# Find optimal number of clusters
clustering.find_elbow(max_k=10)
clustering.plot_elbow()

# Perform clustering
clustering.fit(n_clusters=4)
clustering.plot("feature_1", "feature_2")

# Get results
results = clustering.get_results()
```

#### Standard Scaling
```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and scale data
df = pd.read_csv("your_data.csv")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_columns])
```

## Advanced Features

### Data Preprocessing
- Automated handling of categorical variables
- Support for custom scaling factors
- Built-in data validation and cleaning
- Flexible input format support

### Visualization
- Automated elbow curve generation
- Cluster visualization with customizable parameters
- Support for various plot types and color schemes
- Interactive plotting capabilities

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by scikit-learn's clustering implementations
- Built with best practices from the machine learning community
- Developed to support both educational and professional use cases