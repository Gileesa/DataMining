import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Parameter for Lasso regression
alpha_value = 0.1

df = pd.read_csv("DataMining/csv_files/mood_window_dataset.csv")
print(df.head())

# Prepare the data for feature selection
def mood_to_classification(mood):

    if mood <= 3:
        return 'low'
    elif mood > 3 and mood <= 6:
        return 'medium'
    else:
        return 'high'

df['target_classification'] = df['target'].apply(mood_to_classification)

X = df.drop(columns=['id', 'date', 'target', 'target_classification'])
y = df['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform feature selection using Lasso regression
lasso = Lasso(alpha=alpha_value)
lasso.fit(X_scaled, y)

# Get the feature importance scores
feature_importance = pd.Series(lasso.coef_, index=X.columns)
print("Feature importance scores:")     
print(feature_importance.sort_values(ascending=False))

# Store the selected features and its coefficients
selected_features = []
for feature, importance in feature_importance.items():
    if importance != 0:
        selected_features.append(feature)

print("\nSelected features:")
print(selected_features)

# Create a new DataFrame with the selected features
X_selected = df[selected_features]
print("\nData with selected features:")
print(X_selected.head())

# Save the selected features to a new CSV file
selected_df = df[['id', 'date', 'target', 'target_classification'] + selected_features]
selected_df.to_csv("DataMining/csv_files/mood_selected_features.csv", index=False)