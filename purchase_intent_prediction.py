import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import time
warnings.filterwarnings('ignore')

# Start timing
start_time = time.time()

# Load the data
print("Loading data...")
events_df = pd.read_csv('data/events.csv')
item_props_df1 = pd.read_csv('data/item_properties_part1.csv')
item_props_df2 = pd.read_csv('data/item_properties_part2.csv')

# Combine item properties
item_props_df = pd.concat([item_props_df1, item_props_df2], ignore_index=True)

# Filter out non-numeric property values
print("Filtering numeric properties...")
item_props_df['property'] = pd.to_numeric(item_props_df['property'], errors='coerce')
item_props_df = item_props_df.dropna(subset=['property'])

# Calculate unique properties count per item
print("Calculating unique properties...")
unique_props = item_props_df.groupby('itemid')['property'].nunique().reset_index()
unique_props.columns = ['itemid', 'unique_properties_count']

# Convert timestamps to datetime
events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
item_props_df['timestamp'] = pd.to_datetime(item_props_df['timestamp'])

# Calculate dwell time (time between first and last view of a product by a user)
print("Calculating dwell time...")
user_item_times = events_df.groupby(['visitorid', 'itemid']).agg({
    'timestamp': ['min', 'max']
}).reset_index()
user_item_times.columns = ['visitorid', 'itemid', 'first_view', 'last_view']
user_item_times['dwell_time'] = (user_item_times['last_view'] - user_item_times['first_view']).dt.total_seconds()
user_item_times['dwell_time'] = user_item_times['dwell_time'].fillna(0)  # Fill NaN with 0 for single views

# Create target variable (1 for transaction, 0 for view/addtocart)
events_df['target'] = (events_df['event'] == 'transaction').astype(int)

# Merge events with item properties (excluding value column)
print("Merging data...")
merged_df = pd.merge_asof(
    events_df.sort_values('timestamp'),
    item_props_df[['timestamp', 'itemid', 'property']].sort_values('timestamp'),
    on='timestamp',
    by='itemid',
    direction='backward'
)

# Merge with dwell time and unique properties
merged_df = merged_df.merge(user_item_times[['visitorid', 'itemid', 'dwell_time']], 
                          on=['visitorid', 'itemid'], 
                          how='left')
merged_df = merged_df.merge(unique_props, on='itemid', how='left')

def engineer_features(merged_df):
    # total_events and total_purchases_x (per user)
    user_agg = merged_df.groupby('visitorid').agg({
        'event': 'count',
        'target': 'sum',
        'dwell_time': 'mean'
    }).reset_index()
    user_agg.columns = ['visitorid', 'total_events', 'total_purchases_x', 'avg_dwell_time_user']
    merged_df = merged_df.merge(user_agg, on='visitorid', how='left')

    # total_views, total_purchases_y, avg_property, avg_dwell_time_product, unique_properties_count (per item)
    item_agg = merged_df.groupby('itemid').agg({
        'event': 'count',
        'target': 'sum',
        'property': 'mean',
        'dwell_time': 'mean',
        'unique_properties_count': 'first'
    }).reset_index()
    item_agg.columns = ['itemid', 'total_views', 'total_purchases_y', 'avg_property', 'avg_dwell_time_product', 'unique_properties_count']
    merged_df = merged_df.merge(item_agg, on='itemid', how='left')

    # Ensure all required columns exist
    required = [
        'total_purchases_x', 'total_purchases_y', 'total_events', 'dwell_time',
        'avg_dwell_time_user', 'avg_dwell_time_product', 'total_views',
        'unique_properties_count', 'avg_property', 'property'
    ]
    for col in required:
        if col not in merged_df.columns:
            merged_df[col] = 0
    return merged_df[required]

# Feature engineering
print("Engineering features...")
X = engineer_features(merged_df).fillna(0)
y = merged_df['target']

# # Apply Isolation Forest for outlier detection
# print("Applying Isolation Forest...")
# iso_forest = IsolationForest(contamination=0.1, random_state=42)
# outlier_scores = iso_forest.fit_predict(X)
# X['is_outlier'] = outlier_scores

# # Remove outliers from the dataset
# X = X[outlier_scores == 1]
# y = y[outlier_scores == 1]

# save the merged_df to a csv file
merged_df.to_csv('data/merged_df.csv', index=False)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the model
print("Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print evaluation metrics
print("\nModel Evaluation:")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("Average Precision Score:", average_precision_score(y_test, y_pred_proba))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

def plot_results(model, X, y_test, y_pred_proba):
    # --- Feature Importance Plot ---
    plt.figure(figsize=(8, 5))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features_sorted = [X.columns[i] for i in indices]
    plt.bar(range(len(importances)), importances[indices], color="skyblue")
    plt.xticks(range(len(importances)), features_sorted, rotation=45, ha='right')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()

    # --- ROC Curve Plot ---
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()

    # --- Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

# Generate and save plots
plot_results(model, X_test, y_test, y_pred_proba)

# Print runtime
end_time = time.time()
runtime = end_time - start_time
print(f"\nTotal runtime: {runtime:.2f} seconds") 