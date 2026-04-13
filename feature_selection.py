import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

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
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Get the feature importance scores
feature_importance = pd.Series(lasso.coef_, index=X.columns)
print("Feature importance scores:")     
print(feature_importance.sort_values(ascending=False))

selected_features = []
for feature, importance in feature_importance.items():
    if importance != 0:
        selected_features.append(feature)

print("\nSelected features:")
print(selected_features)

X_selected = df[selected_features]
print("\nData with selected features:")
print(X_selected.head())
