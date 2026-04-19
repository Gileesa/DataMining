# DataMining
For the VU course Data Mining Techniques (X_400108)

This project is part of the Data Mining Techniques course and focuses on predicting the next-day mood of smartphone users based on behavioral data.

The dataset contains time-stamped measurements (e.g., mood, screen time, activity, app usage) for multiple users. After cleaning and preprocessing the data, a window-based dataset is created using recent history (e.g., past 5 days) to predict future mood.

Pipeline
Data Cleaning & Imputation
Missing values are handled using interpolation and KNN-based imputation. Outliers are removed and replaced using interpolation.

Feature Engineering (Task 1C)
A window-based dataset is constructed where features represent aggregated values (e.g., averages) over previous days, and the target is the next-day mood.

Classification Dataset Preparation
The continuous mood score is converted into three classes:

Low (<= 6)
Medium (6 < mood <= 7)
High (> 7)

The dataset is split into train and test sets using a time-aware split per user to avoid data leakage.

Modeling (Task 2A)




Structure
data_cleaning.py → cleaning + window dataset creation
classification_dataset.py → class creation + train/test split
data_loader.py → loading prepared datasets
random_forest.py → classification model
config.py → shared parameters