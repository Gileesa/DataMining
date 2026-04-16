#
#
#

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.impute import KNNImputer
from data_exploration import print_general_info



# ===== GLOBAL VARS ========
WINDOW_SIZE:int = 5 #days
TRIMMING:bool = False

RELEVANT_FEATURES = ['mood', 'screen', 'circumplex.valence', 'circumplex.arousal', 'activity', 'appCat.social', 'appCat.entertainment', 'appCat.communication', 'appCat.other', 'appCat.travel']




def linear_interpol(x):
    ''' does (previous_point + next_point)/2 only when both exist '''
    return x.interpolate(method='linear', limit_area='inside')

def backward_interpol(x):
    ''' just fill in next value'''
    return x.bfill()

def missing_imputation(df, varname: str, imputation_func):
    ''' 
    This function will impute missing values for the 'varname' under the [value] column.
    This is done through some imputation func, e.g (previous_point + next_point)/2
    Mostly used for the varnames: circumplex.valence and circumplex.arousal
    '''
    
    new_df = df.copy()

    # filter only the variable of interest
    mask = new_df['variable'] == varname

    # sort by id and date to ensure correct interpolation order
    subset = new_df.loc[mask].sort_values(['id', 'date'])

    # apply interpolation per id
    subset['value'] = subset.groupby('id')['value'].transform(
        lambda x: imputation_func(x)
    )

    # put the interpolated values back
    new_df.loc[mask, 'value'] = subset['value']

    return new_df

def remove_extremes(df, varname:str='screen', min_value:float=0, max_value:float=700):
    '''
    Function that removes extreme values, both at min and max.
    E.g remove screentime below zero and above 700 minutes.
    Removed datapoints will be imputed through linear interpolation

    Params:
    - df: dataframe that we will clean
    - varname: name of variable we are cleaning (e.g 'screen' for screentime)
    - min_value: minimal value for the variable we are cleaning
    - max_value: maximal value for the variable we are cleaning
    Returns:
    - new_df: the cleaned dataframe with the extremes replaced by linearly interpolated datapoints
    '''

    new_df = df.copy()
    mask = new_df['variable'] == varname

    values = new_df.loc[mask, 'value']

    extreme_mask = (values > max_value) | (values < min_value)

    new_df.loc[mask, 'value'] = values.mask(extreme_mask)

    new_df.loc[mask, 'value'] = (
        new_df.loc[mask]
        .groupby('id')['value']
        .transform(lambda x: x.interpolate())
    )

    return new_df

def plot_histogram(df, varname:str, uid:str=None):
    ''' 
    Plots a histogram of a specific variable to see its distribution of values.
    Also shows min and max value in plots.
    Can be used to observe outliers, as well as overall distribution (e.g normal)

    Params:
    - df: dataframe we are plotting from
    - varname: variable name we want to plot histogram of
    - uid: user id; fill if you want to plot values for one specific user
    Returns:
    - None
    '''
    data = df['value'].dropna()

    min_val = data.min()
    max_val = data.max()

    plt.figure(figsize=(8,5))
    plt.hist(data, bins=15, color='lightpink', edgecolor='red')

    # title with varname + user id
    title = f"Histogram of {varname}"
    if uid is not None:
        title += f" (ID: {uid})"

    plt.title(title)
    plt.xlabel(varname)
    plt.ylabel('Frequency')

    plt.text(
        0.95, 0.95,
        f"Min: {min_val:.2f}\nMax: {max_val:.2f}",
        transform=plt.gca().transAxes,
        ha='right',
        va='top',
        bbox=dict(facecolor='white', alpha=0.7)
    )

    plt.show()

    print(f"\n{title} statistics:")
    print("Mean:", data.mean())
    print("Median:", data.median())
    print("Min:", min_val)
    print("Max:", max_val)


def plot_KNN(user_full, user_imputed, uid, varnames):
    for var in varnames:
        plt.figure(figsize=(10, 5))

        original = user_full[var]
        imputed = user_imputed[var]

        plt.plot(imputed.index, imputed, '.-')
        plt.plot(original.index, original, 'o', label=f'{var} (original)', alpha=0.6)

        # highlight imputed points
        imputed_mask = original.isna()
        plt.scatter(
            imputed.index[imputed_mask],
            imputed[imputed_mask],
            color='red',
            label='imputed points'
        )

        plt.title(f'KNN Imputation - {var} (ID={uid})')
        plt.xlabel('Date')
        plt.ylabel(var)
        plt.legend()
        plt.savefig(f'Figures/KNN/KNN_imputation_{uid}_{var}.png')
        # plt.show()
        plt.close()


def apply_KNN_imputation(
        df,
        neighbours:int=3,
        relevant_features = None, # put None for all features
        plotting:bool=True
):
    imputer = KNNImputer(n_neighbors=neighbours)

    if relevant_features is not None:
        varnames = relevant_features
    else:
        varnames = list(df['variable'].unique())

    df = df[df['variable'].isin(varnames)].copy()
    df['date_day'] = df['date'].dt.floor('D')
    cleaned_dfs = []
    for uid, user_df in df.groupby('id'):
        # compute daily average and std
        # we pivot here
        user_daily = user_df.pivot_table(
            index='date_day',
            columns='variable',
            values='value',
            aggfunc='mean'
        ).sort_index()

        varnames_user = user_daily.columns.tolist()

        if user_daily.empty:
            # skip rest of loop
            continue 

        # find range of dates
        full_range = pd.date_range(
            user_daily.index.min(),
            user_daily.index.max(),
            freq='D'
        )
        user_full = user_daily.reindex(full_range)

        # impute missing values - KNN
        user_imputed = pd.DataFrame(
            imputer.fit_transform(user_full),
            index=user_full.index,
            columns=user_full.columns
        )

        # unpivot (needed for rest of pipeline)
        user_melt = user_imputed.reset_index().melt(
            id_vars='index',
            value_vars=varnames_user,
            var_name='variable',
            value_name='value'
        )

        user_melt = user_melt.rename(columns={'index': 'date'})
        user_melt['id'] = uid

        # print imputations for debug
        for var in varnames_user:
            imputed_mask = user_full[var].isna() & user_imputed[var].notna()
            for date, val in user_imputed.loc[imputed_mask, var].items():
                print(f"[KNN IMPUTED] ID={uid} | var={var} | date={date.date()} | value={val:.3f}")

        if plotting and uid=="AS14.01":
            plot_KNN(user_full, user_imputed, uid, varnames_user)

        cleaned_dfs.append(user_melt)

    result = pd.concat(cleaned_dfs, ignore_index=True)
    return result



def plot_cleaned_per_id(df_clean, original_df, feature: str = 'mood', save_path=None):
    """
    Plot the cleaned time series per ID for a given feature.
    Compatible with the melted output of apply_KNN_imputation.

    params:
    - df_clean:     melted DataFrame with columns ['id', 'date', 'variable', 'value']
                    (output of apply_KNN_imputation)
    - original_df:  DataFrame with raw data, must contain ['id', 'date', 'variable', 'value']
    - feature:      which variable to plot (must be in 'variable' column)
    - save_path:    optional path to save the figure
    returns:
    - None
    """
    # filter both dfs to the feature of interest
    df_feat = df_clean[df_clean['variable'] == feature].copy()
    original_feat = original_df[original_df['variable'] == feature].copy()

    # normalise dates to day-level
    df_feat['date'] = pd.to_datetime(df_feat['date']).dt.floor('D')
    original_feat['date'] = pd.to_datetime(original_feat['date']).dt.floor('D')

    ids = df_feat['id'].unique()
    ncols = 6
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3), sharey=True)
    axes = axes.flatten()

    print(f'\n\n ==== REMOVED DATES INFO {save_path} =======')
    for i, uid in enumerate(ids):
        ax = axes[i]

        user_clean = df_feat[df_feat['id'] == uid]
        user_orig  = original_feat[original_feat['id'] == uid]

        original_dates = set(user_orig['date'].unique())
        cleaned_dates  = set(user_clean['date'].unique())

        n_original = len(original_dates)
        n_cleaned  = len(cleaned_dates)
        n_removed  = len(original_dates - cleaned_dates)
        n_added    = len(cleaned_dates - original_dates)

        print(f"- ID {uid}: {n_original} original days → {n_cleaned} cleaned days "
              f"| removed (tail/head) {n_removed}, added (KNN) {n_added}")

        user_clean_sorted = user_clean.sort_values('date')
        ax.plot(user_clean_sorted['date'], user_clean_sorted['value'],
                marker='o', linestyle='-', color='gold', linewidth=1.2, markersize=3)
        ax.set_title(str(uid), fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)

    print('number of ids: ', len(ids))
    for j in range(len(ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Cleaned {feature} per ID', fontsize=12, y=0.99)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()



def create_one_size_window(
    df,
    feature_names=['screen', 'mood'],
    window_size=5,
    target_feature='mood',
    save_path='csv_files/window_dataset.csv'
):

    merged_df = None

    for feature in feature_names:
        # debug
        ids_with_feature = df[df['variable'] == feature]['id'].unique()
        print(f"{feature}: {len(ids_with_feature)}/27 users")

        # rename columns (temporarily)
        temp = df[df['variable'] == feature][['id', 'date', 'value']].rename(columns={'value': feature})

        if merged_df is None:
            # if nothing to merge into, become temp
            merged_df = temp
        else:
            # merge dataframes
            merged_df = pd.merge(
                merged_df,
                temp,
                on=['id', 'date'],
                how='inner' # keep only dates where all features exist
            )

    merged_df = merged_df.sort_values(['id', 'date'])

    all_rows = []

    # create windows 
    for uid, user_df in merged_df.groupby('id'):
        user_df = user_df.sort_values('date').reset_index(drop=True)

        if len(user_df) < window_size + 1:
            # skip if no target
            continue

        for i in range(window_size, len(user_df)):
            row = {}

            # id & date
            row['id'] = uid
            row['date'] = user_df.loc[i, 'date']

            # average over full window instead of one-day values
            window_slice = user_df.iloc[i - window_size:i]

            for feature in feature_names:
                row[f'{feature}_avg'] = window_slice[feature].mean()

            # target (e.g mood)
            row['target'] = user_df.loc[i, target_feature]

            all_rows.append(row)

    dataset = pd.DataFrame(all_rows)

    dataset.to_csv(save_path, index=False)
    print(f"\nSaved dataset to: {save_path}")
    print(f"Shape: {dataset.shape}")

    return dataset


### ==== MAIN CLEANING PART ========
df = pd.read_csv('dataset_mood_smartphone.csv')

#  make datetime
df['date'] = pd.to_datetime(df['time'], errors='coerce')

#====== add missing row values =======
cleaned_df = missing_imputation(df, 'circumplex.valence', linear_interpol)
cleaned_df = missing_imputation(cleaned_df, 'circumplex.arousal', linear_interpol)
# still one value missing in circumplex.valence; try backward interpol
cleaned_df = missing_imputation(cleaned_df, 'circumplex.valence', backward_interpol)
print_general_info(cleaned_df)

#==== remove extreme values, replace with neighbour average (linear interpol) ======
cleaned_df = remove_extremes(cleaned_df, varname='screen', max_value=1000)
cleaned_df = remove_extremes(cleaned_df, varname='appCat.social', max_value=1000)
cleaned_df = remove_extremes(cleaned_df, varname='appCat.communication', max_value=1000)
cleaned_df = remove_extremes(cleaned_df, varname='appCat.entertaiment', max_value=1000)
cleaned_df = remove_extremes(cleaned_df, varname='appCat.other', max_value=1000)  
cleaned_df = remove_extremes(cleaned_df, varname='appCat.travel', max_value=1000)

#====== add missing dates =========
print('DOING KNN imputation')
KNN_impute = apply_KNN_imputation(cleaned_df, 3, RELEVANT_FEATURES)
plot_cleaned_per_id(KNN_impute, cleaned_df, save_path='Figures/KNN/dates_mood.png')
plot_cleaned_per_id(KNN_impute, cleaned_df, feature='screen', save_path='Figures/KNN/dates_screen.png')
plot_cleaned_per_id(KNN_impute, cleaned_df, feature='circumplex.valence', save_path='Figures/KNN/dates_valence.png')
plot_cleaned_per_id(KNN_impute, cleaned_df, feature='circumplex.arousal', save_path='Figures/KNN/dates_arousal.png')
plot_cleaned_per_id(KNN_impute, cleaned_df, feature='activity', save_path='Figures/KNN/dates_activity.png')

# Create window, averaging n days
df_windows = create_one_size_window(
    KNN_impute,
    feature_names=RELEVANT_FEATURES,
    window_size=5,
    save_path='csv_files/KNN/KKN_one_size_mood_window_dataset.csv'
)

print('WINDOW number of ids: ', len(df_windows['id'].unique()))