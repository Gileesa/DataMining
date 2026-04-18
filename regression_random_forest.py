from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import os 

from data_loader import load_data
from config import N_SPLITS, RANDOM_STATE

os.makedirs("DataMining/Figures/Regression/RandomForest", exist_ok=True)
os.makedirs("DataMining/csv_files/Regression/RandomForest", exist_ok=True)


# Load data
train_df, test_df, X_train, X_test, feature_cols = load_data()

y_train = train_df['target']
y_test  = test_df['target']

# TimeSeriesSplit for cross-validation - time-aware splitting
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

rf_param_grid = {"n_estimators"             : [100, 200, 300, 400, 500],                  #nr of trees
                 "criterion"                : ['squared_error', 'absolute_error'],              #how split quality is measured for regression
                 "max_depth"                : [5, 10, 15, 20, None],                    #how deep trees can grow
                 "min_samples_split"        : [2, 4, 6, 8],                  
                 "min_samples_leaf"         : [1, 3, 5, 7],                                  #controls overfitting
                #  "min_weight_fraction_leaf" : [0.0], 
                 "max_features"             : ["sqrt", "log2", 0.5, 1.0],                       #how many features each split can consider - Smaller values → more randomness → better generalization
                #  "min_impurity_decrease"    : [0.0], 
                #  "bootstrap"                : [True], 
                #  "oob_score"                : [False], 
                #   "max_leaf_nodes"           : [None],                                           # Limits number of leaf nodes 
                #  "n_jobs"                   : [None], 
                #  "random_state"             : [None], 
                #  "verbose"                  : [0], 
                #  "warm_start"               : [False], 
                #  "ccp_alpha"                : [0.0], 
                #  "max_samples"              : [None], 
                #  "monotonic_cst"            : [None],
                 
                 }

# Random Search

rf_search = RandomizedSearchCV(
                                estimator           = RandomForestRegressor(random_state= RANDOM_STATE), 
                                param_distributions = rf_param_grid, 
                                n_iter              = 100, 
                                scoring             = "neg_mean_squared_error", 
                                n_jobs              = -1, 
                                refit               = True, 
                                cv                  = tscv, 
                                verbose             = 1, 
                                pre_dispatch        = '2*n_jobs', 
                                random_state        = RANDOM_STATE, 
                                error_score         = "raise", 
                                return_train_score  = False,


                            )

rf_search.fit(X_train, y_train)

print("\n RANDOM SEARCH RESULTS")
print("Best parameters:")
for k, v in rf_search.best_params_.items():
    print(f"{k}: {v}")

    
print(f"Best CV score (neg MSE): {rf_search.best_score_:.4f}")
print(f"Best CV MSE: {-rf_search.best_score_:.4f}")
print("Best estimator:")
print(rf_search.best_estimator_)

best_rf = rf_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

test_mse = mean_squared_error(y_test, y_pred_rf)
test_mae = mean_absolute_error(y_test, y_pred_rf)

print("\n Test RESULTS")
print(f"MSE: {test_mse:.4f}")
print(f"MAE: {test_mae:.4f}")

# Save predictions
results_df = pd.DataFrame({
                            "id"                : test_df["id"].values,
                            "date"              : test_df["date"].values,
                            "actual_value"      : y_test.values,
                            "predicted_value"   : y_pred_rf
                        })

results_df.to_csv(
                    "DataMining/csv_files/Regression/RandomForest/rf_tuned_predictions.csv",
                    index=False
                )

# Save summary
summary_df = pd.DataFrame({
                            "Model"             : ["Random Forest Regressor (baseline)"],
                            "Test MSE"          : [test_mse],
                            "Test MAE"          : [test_mae],
                            "Best Params"       : [str(rf_search.best_params_)],
                            "Best CV Neg MSE"   : [rf_search.best_score_],
                            "Best CV MSE"       : [-rf_search.best_score_],
                        })

summary_df.to_csv(
                    "DataMining/csv_files/Regression/RandomForest/rf_tuned_results.csv",
                    index=False
                )

feat_imp = pd.Series(best_rf.feature_importances_, index=feature_cols)

feat_imp = feat_imp.sort_values()

feat_imp.plot(kind='barh', figsize=(7, 6), title="RF Feature Importances")

plt.tight_layout()
plt.savefig("DataMining/Figures/Regression/RandomForest/rf_feature_importance.png", dpi=150)
plt.savefig("DataMining/Figures/Regression/RandomForest/rf_feature_importance.pdf")
plt.close()