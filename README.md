# DataMining
*For the VU course Data Mining Techniques (X_400108)*

This project is part of the Data Mining Techniques course and focuses on predicting the next-day mood of smartphone users based on behavioral data.

The dataset contains time-stamped measurements (e.g., mood, screen time, activity, app usage) for multiple users. After cleaning and preprocessing the data, a window-based dataset is created using recent history (e.g., past 5 days) to predict future mood.

---

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Pipeline](#pipeline)
- [Structure](#structure)
- [Authors](#authors)

---

## Requirements
Requirements can be found in the requirements.txt file

## Installation
To run this code, run the following commands in the terminal:
> ```
> git clone https://github.com/Gileesa/DataMining.git
> cd datamining
> pip install -r requirements.txt
> ```

In addition, it is important to note that the raw data file 'dataset_mood_smartphone.csv' is **NOT** included in this repository. It must be added to the DataMining folder by the user. Without this csv file, the data exploration part (data_exploration.py) and the data cleaning part (data_cleaning.py) cannot run properly.

## How to Run
This repository consists of many different sub parts, most of which can be run separate from the others. A full overview of all files and their function can be found below under 'Structure'.

Example on how to run the data cleaning part:
> ```
> python data_cleaning.py
> ```

---

## Pipeline

### 1. Data Cleaning & Imputation
Missing values are handled using interpolation and KNN-based imputation. Outliers are removed and replaced using interpolation.

### 2. Feature Engineering *(Task 1C)*
A window-based dataset is constructed where features represent aggregated values (e.g., averages) over previous days, and the target is the next-day mood.

### 3. Classification Dataset Preparation
The continuous mood score is converted into three classes:

| Class | Condition |
|-------|-----------|
| Low | mood ≤ 6 |
| Medium | 6 < mood ≤ 7 |
| High | mood > 7 |

The dataset is split into train and test sets using a time-aware split per user to avoid data leakage.

### 4. Modeling *(Task 2A)*

---

## Structure

| File | Description |
|------|-------------|
| `data_cleaning.py` | Cleaning + window dataset creation |
| `classification_dataset.py` | Class creation + train/test split |
| `Classification_dataset_RNN.py` | |
| `data_loader.py` | Loading prepared datasets |
| `data_exploration.py` | Used for data exploration; not necessary to run pipeline|
| `random_forest.py` | Classification model |
| `regression_random_forest.py` | |
| `plotting_rf_classification.py` | |
| `plotting_rf_regression.py` | |
| `classification_RNN_model.py` | |
| `regression_RNN_model.py` | |
| `RNN_utils.py` | |
| `classification_test.py` | |
| `regression_test.py` | |
| `test_and_train.py` | |
| `xgboost_model.py` | |
| `gridsearch.py` | |
| `plot.py` | |
| `plot_regression.py` | |
| `Comparison.py` | |
| `config.py` | Shared parameters |
| `dataset_mood_smartphone.csv` | Raw dataset (not actually present in this repository)|
| `csv_files/` |Folder for cleaned datasets|
| `Figures/` |Folder for figures|

---

## Authors
Gileesa McCormack, Sooriyaa Karunaharan, Rohan Kanhai

Group 154, Data Mining Techniques (X_400108)

Vrije Universiteit Amsterdam
