# Decision Tree Crowdfunding Predictor

This project implements a decision tree classifier to predict crowdfunding outcomes. It uses scikit-learn to process crowdfunding data, train a decision tree model, and visualize the results.

## Features

- Data preprocessing using pandas and sklearn
- Decision tree model implementation
- Model evaluation with accuracy metrics
- Decision tree visualization using pydotplus
- Exports tree visualization in both PDF and PNG formats

## Project Structure

```
Decision_Tree/
│
├── data/
│   └── crowdfunding_data.csv
│
├── graphs/
│   ├── crowdfunding_tree.pdf
│   └── crowdfunding_tree.png
│
└── predict_decision_tree_example.py
```

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```
python predict_decision_tree_example.py
```

The script will:
1. Load and preprocess the crowdfunding data
2. Train a decision tree classifier
3. Make predictions and evaluate the model
4. Generate and save tree visualizations

## Credits

This project was developed as part of the Machine Learning & AI Micro Boot Camp at Arizona State University (2023). The code represents class work completed during the course.

## License

This project is provided for educational purposes only. All rights reserved.