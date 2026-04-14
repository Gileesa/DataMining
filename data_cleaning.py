#
#
#

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_exploration import print_general_info

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

def remove_extremes(df, varname='screen', min_value=0, max_value=700):

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

def plot_histogram(df, varname, uid=None):
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


def clean_time_series(
    df,
    varname='mood',
    max_gap=3, #max gap of days
    smooth=False,
    plotting: bool=False
):
    '''
    Function that can impute missing date values for 'varname'. 
    In addition, this function cuts away parts on the head and tail of the timeseries
    that have a gap of >3 days with the 'main body' of the timeseries.
    '''
    df = df[df['variable'] == varname].copy()
    df['date_day'] = df['date'].dt.floor('D')
    cleaned_dfs = []
    for uid, user_df in df.groupby('id'):

        #compute daily average and std
        user_daily = (
            user_df
            .groupby('date_day')['value']
            .agg(value='mean', value_std='std')
            .sort_index()
        )
        if user_daily.empty:
            # skip rest of loop
            continue 
        elif plotting:
            # plot histogram of values to see range
            plot_histogram(user_daily, varname, uid)

        # find range of dates
        full_range = pd.date_range(
            user_daily.index.min(),
            user_daily.index.max(),
            freq='D'
        )
        user_full = user_daily.reindex(full_range)

        # trim head and tail of data if there is a large gap
        is_valid = ~user_full['value'].isna()
        valid_idx = np.where(is_valid)[0] #find dates with data

        if len(valid_idx) == 0:
            continue

        # find distance (data-wise) between existing data points
        distance_between_data = np.diff(valid_idx) 
        split_points = np.where(distance_between_data > max_gap)[0]

        # split data in continuous chunks
        # split at positions in split_points
        segments = []
        start = 0
        for sp in split_points:
            segments.append(valid_idx[start:sp+1])
            start = sp + 1
        segments.append(valid_idx[start:])

        # trim isolated head/tail segments, keep everything in between 
        # only remove the first segment if it is separated from the rest by a large gap 
        # only remove the last segment if it is separated from the rest by a large gap
        largest_segment = max(segments, key=len)

        start_idx = largest_segment[0]
        end_idx = largest_segment[-1]

        user_trimmed = user_full.iloc[start_idx:end_idx+1]

        # impute missing values (mean only — std stays NaN for imputed days)
        user_imputed = user_trimmed.copy()
        user_imputed['value'] = user_trimmed['value'].interpolate(method='time')

        # print imputations for debug
        imputed_mask = user_trimmed['value'].isna() & user_imputed['value'].notna()
        for date, val in user_imputed.loc[imputed_mask, 'value'].items():
            print(f"[IMPUTED] ID={uid} | date={date.date()} | value={val:.3f}")

        # smoothing
        # since data is spiky, we can make all values be the average of its neighbours
        # this reduces noise, but means loss of data
        if smooth:
            user_imputed['value'] = user_imputed['value'].rolling(
                window=3,
                center=True,
                min_periods=1
            ).mean()

        # create df
        clean_df = pd.DataFrame({
            'id': uid,
            'date': user_imputed.index,
            'value': user_imputed['value'].values,
            'value_std': user_imputed['value_std'].values  # NaN for imputed days
        })
        cleaned_dfs.append(clean_df)
    result = pd.concat(cleaned_dfs, ignore_index=True)
    return result


def plot_cleaned_per_id(df_clean, original_df, value_col='value', feature:str='mood', save_path=None):
    """
    Plot the cleaned time series per ID and prints how many days were removed (tail/head) and added (interpolation)
    params:
    - df_clean: DataFrame with columns ['id', 'date', value_col]
    - original_df: DataFrame with raw data, must contain ['id', 'date', 'variable']
    - value_col: column name containing the numeric value to plot
    - save_path: optional path to save the figure
    returns:
    - None
    """
    ids = df_clean['id'].unique()
    ncols = 6
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3), sharey=True)
    axes = axes.flatten()

    # filter original to mood only, normalize dates to day
    original_mood = original_df[original_df['variable'] == feature].copy()
    original_mood['date'] = pd.to_datetime(original_mood['date']).dt.floor('D')
    df_clean['date'] = pd.to_datetime(df_clean['date']).dt.floor('D')

    print(f'\n\n ==== REMOVED DATES INFO {save_path} =======')
    for i, uid in enumerate(ids):
        ax = axes[i]
        user_clean = df_clean[df_clean['id'] == uid]
        user_orig  = original_mood[original_mood['id'] == uid]

        original_dates = set(user_orig['date'].unique())
        cleaned_dates  = set(user_clean['date'].unique())

        n_original = len(original_dates)
        n_cleaned  = len(cleaned_dates)
        n_removed  = len(original_dates - cleaned_dates)
        n_added    = len(cleaned_dates - original_dates)

        print(f"- ID {uid}: {n_original} original days → {n_cleaned} cleaned days "
              f"| removed (tail/head) {n_removed}, added (interpol) {n_added}")

        user_clean_sorted = user_clean.sort_values('date')
        ax.plot(user_clean_sorted['date'], user_clean_sorted[value_col],
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




def create_window_dataset_from_clean(
    dfs,                # list of dataframes
    feature_names=['screen', 'mood'],      # list of feature names (same order)
    window_size=5,
    target_feature='mood',
    save_path='csv_files/window_dataset.csv'
):
    """
    Generalized sliding window dataset for multiple features.

    params:
    - dfs: list of cleaned dataframes
    - feature_names: list of feature names (same length as dfs)
    - target_feature: which feature to predict (default = mood)
    """

    merged_df = None

    for df, feature in zip(dfs, feature_names):
        temp = df[['id', 'date', 'value']].rename(columns={'value': feature})

        if merged_df is None:
            merged_df = temp
        else:
            merged_df = pd.merge(
                merged_df,
                temp,
                on=['id', 'date'],
                how='inner'
            )

    merged_df = merged_df.sort_values(['id', 'date'])

    all_rows = []

    # create windows 
    for uid, user_df in merged_df.groupby('id'):
        user_df = user_df.sort_values('date').reset_index(drop=True)

        if len(user_df) < window_size + 1:
            continue

        for i in range(window_size, len(user_df)):
            row = {}

            # id + date
            row['id'] = uid
            row['date'] = user_df.loc[i, 'date']

            # loop over all features
            for feature in feature_names:
                for j in range(window_size):
                    row[f'{feature}_t-{window_size-j}'] = user_df.loc[
                        i - window_size + j, feature
                    ]

            # target
            row['target'] = user_df.loc[i, target_feature]

            all_rows.append(row)

    dataset = pd.DataFrame(all_rows)

    # save to csv
    dataset.to_csv(save_path, index=False)
    print(f"\nSaved dataset to: {save_path}")
    print(f"Shape: {dataset.shape}")

    return dataset



df = pd.read_csv('dataset_mood_smartphone.csv')
#  make datetime
df['date'] = pd.to_datetime(df['time'], errors='coerce')

#====== add missing row values =======
cleaned_df = missing_imputation(df, 'circumplex.valence', linear_interpol)
cleaned_df = missing_imputation(cleaned_df, 'circumplex.arousal', linear_interpol)
# still one value missing in circumplex.valence; try backward interpol
cleaned_df = missing_imputation(cleaned_df, 'circumplex.valence', backward_interpol)
print('\n\n====== IMPUTED MISSING ROW VALUES HEAD =======\n', cleaned_df.head(20))
print(cleaned_df.head(20))
print_general_info(cleaned_df)

#==== remove extreme values, replace with neighbour average (linear interpol) ======
cleaned_df = remove_extremes(cleaned_df, varname='screen', max_value=700)
cleaned_df = remove_extremes(cleaned_df, varname='appCat.social', max_value=1000)
cleaned_df = remove_extremes(cleaned_df, varname='appCat.game', max_value=1000)
cleaned_df = remove_extremes(cleaned_df, varname='appCat.entertaiment', max_value=1000)

#====== add missing dates =========
# + cut off parts of timeseries with large gaps
clean_dates_mood = clean_time_series(cleaned_df)
clean_dates_screentime = clean_time_series(cleaned_df, varname = 'screen')
clean_dates_valence = clean_time_series(cleaned_df, varname = 'circumplex.valence')
clean_dates_arousal = clean_time_series(cleaned_df, varname = 'circumplex.arousal')
clean_dates_activity = clean_time_series(cleaned_df, varname = 'activity')
clean_dates_social = clean_time_series(cleaned_df, varname = 'appCat.social')
clean_dates_game = clean_time_series(cleaned_df, varname = 'appCat.game')
clean_dates_entertainment = clean_time_series(cleaned_df, varname = 'appCat.entertainment')

print('\n\n====== IMPUTED DATES HEAD =======\n', clean_dates_mood.head(20))
plot_cleaned_per_id(clean_dates_mood, cleaned_df, save_path='Figures/dates_mood.png')
plot_cleaned_per_id(clean_dates_screentime, cleaned_df, feature='screen', save_path='Figures/dates_screen.png')
plot_cleaned_per_id(clean_dates_valence, cleaned_df, feature='circumplex.valence', save_path='Figures/dates_valence.png')
plot_cleaned_per_id(clean_dates_arousal, cleaned_df, feature='circumplex.arousal', save_path='Figures/dates_arousal.png')
plot_cleaned_per_id(clean_dates_activity, cleaned_df, feature='activity', save_path='Figures/dates_activity.png')


df_windows = create_window_dataset_from_clean(
    [clean_dates_mood, clean_dates_screentime, clean_dates_valence, clean_dates_arousal, clean_dates_activity, clean_dates_social, clean_dates_game, clean_dates_entertainment],
    feature_names=['mood', 'screen', 'circumplex.valence', 'circumplex.arousal', 'activity', 'appCat.social', 'appCat.game', 'appCat.entertainment'],
    window_size=5,
    save_path='csv_files/mood_window_dataset2.csv'
)
print(df_windows.head(20))