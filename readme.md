# Machine Learning Analysis Tools

A collection of machine learning scripts and tools, focusing on clustering analysis and other ML techniques.

## Overview

This repository contains implementations of various machine learning algorithms and analysis tools, with a current focus on clustering analysis. The tools are designed to be both educational and practical for real-world data analysis.

## Scripts

- `simple_clustering_class.py`: Reusable class for clustering analysis with built-in visualization
- `K_means_example.py`: Customer segmentation example using K-means clustering
- `best_K_means_example.py`: Script for finding optimal K value using elbow method

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from simple_clustering_class import SimpleClusterAnalysis

clustering = SimpleClusterAnalysis()
clustering.load_data("your_data.csv")
clustering.find_elbow(max_k=10)
clustering.fit(n_clusters=4)
```

## Prerequisites

- pandas
- numpy
- scikit-learn
- matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details.