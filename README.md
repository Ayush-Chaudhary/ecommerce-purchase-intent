# E-commerce Purchase Intent Prediction

This project predicts purchase intent in e-commerce using the Retail Rocket dataset. It implements a machine learning pipeline that processes user interaction data and predicts whether a user will purchase a product they've viewed.

## Dataset

The project uses the [Retail Rocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) which contains:
- Over 2.7 million user events (views, add-to-cart, transactions)
- 417,053 unique items
- 1,407,580 unique visitors

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place the following files in the `data` directory:
- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`

## Running the Code

1. Ensure your data files are in the correct location:
```
project_root/
├── data/
│   ├── events.csv
│   ├── item_properties_part1.csv
│   └── item_properties_part2.csv
├── purchase_intent_prediction.py
├── requirements.txt
└── README.md
```

2. Run the prediction script:
```bash
python purchase_intent_prediction.py
```

## Output

The script will generate:
1. `feature_importance.png`: Shows the relative importance of each feature
2. `roc_curve.png`: Displays the model's ROC curve with AUC score
3. `confusion_matrix.png`: Visualizes true/false positives and negatives
4. `merged_df.csv`: Contains the processed dataset with engineered features

The console output will show:
- Model evaluation metrics (classification report)
- ROC AUC and Average Precision scores
- Feature importance rankings
- Total runtime of the script

## Features

The model uses several engineered features:
- User-level features: total purchases, total events, average dwell time
- Product-level features: total views, total purchases, average property values
- Interaction features: dwell time, unique properties count

## Model Details

- Algorithm: Random Forest Classifier
- Class imbalance handling: SMOTE (Synthetic Minority Over-sampling)
- Key hyperparameters:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 10
  - min_samples_leaf: 4
  - class_weight: 'balanced'

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- imbalanced-learn >= 0.8.0
- seaborn (for visualization)
- matplotlib (for plotting) 