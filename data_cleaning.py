#
#
#


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('dataset_mood_smartphone.csv')
#  make datetime
df['date'] = pd.to_datetime(df['time'], errors='coerce')

print(df.head(20))


# plot average mood vs average screen time for each [id]
def plot_mood_vs_screentime(df):
    # Create pivot table of retrieve columns 'screen' and 'mood'
    # also takes average value of for each id over all dates
    avg_df = df.pivot_table(index='id', columns='variable', values='value', aggfunc='mean').reset_index()

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(avg_df['screen'], avg_df['mood'], color='blue')
    plt.xlabel('Average Screen Time')
    plt.ylabel('Average Mood')
    plt.title('Average Mood vs Average Screen Time per User')
    plt.grid(True)
    plt.show()


def plot_valence_and_arousal_vs_screentime(df):
    # get average for screen and arousal/valence
    avg_df_arousal = df.pivot_table(index='id', columns='variable', values='value', aggfunc='mean').reset_index()

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(avg_df_arousal['screen'], avg_df_arousal['circumplex.arousal'], color='blue', label="arousal")
    plt.xlabel('Average Screen Time')
    plt.ylabel('Average Arousal')
    plt.title('Average Arousal and Valence vs Average Screen Time per User')
    # plt.legend()
    plt.grid(True)
    plt.show()

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(avg_df_arousal['screen'], avg_df_arousal['circumplex.valence'], color='red', label="valence")
    plt.xlabel('Average Screen Time')
    plt.ylabel('Average Valence')
    plt.title('Average Valence vs Average Screen Time per User')
    # plt.legend()
    plt.grid(True)
    plt.show()


def histogram_of_mood(df):
    mood_df= df[df['variable']=='mood']['value']
    plt.hist(mood_df, bins=15, color='lightpink', edgecolor='red')
    plt.title('Histogram of Mood')
    plt.xlabel('Mood')
    plt.ylabel('Frequency')
    plt.show()
 
    print("Mood statistics:")
    print("Mean:", mood_df.mean())
    print("Median:", mood_df.median())
    print("Min:", mood_df.min())
    print("Max:", mood_df.max())



def histogram_dates(df):

    plt.figure(figsize=(8,5))
    plt.hist(df['date'].dropna(), bins=15, color='lightgreen', edgecolor='black')
    plt.title('Histogram of Dates')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.show()

    # Date statistics
    print("Date statistics:")
    print("Mean:", df['date'].mean())
    print("Median:", df['date'].median())
    print("Min:", df['date'].min())
    print("Max:", df['date'].max())


def print_general_info(df):
    print("\n\n====== GENERAL INFO==========")
    # print all distict options in 'variable'
    distinct_variables = df['variable'].unique().tolist()
    print(distinct_variables)
    
    # Printing general info
    num_ids = df['id'].nunique()
    print("\nNumber of distinct IDs:", num_ids)
    num_days = df['date'].dt.date.nunique()
    print("Number of distinct days:", num_days)
    missing_values = df.isnull().sum()
    print("\nMissing values per column:\n", missing_values)

    # we can see lots of 'value' have missing values
    # find missing rows and count number per variable
    missing_value_rows = df[df['value'].isnull()]
    missing_by_variable = missing_value_rows['variable'].value_counts()
    print("Missing values per variable:")
    print(missing_by_variable) # circumplex.valence: 156, circumplex.arousal: 46
    print("==========================")

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


def plot_valence_per_user_full_calendar(df, varname: str = 'circumplex.valence'):
    valence_df = df[df['variable'] == varname].copy()
    valence_df = valence_df.sort_values(['id', 'date'])
    ids = valence_df['id'].unique()

    ncols = 6
    nrows = int(np.ceil(len(ids) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3), sharey=True)
    axes = axes.flatten()

    print(f"==== MISSING DATE INFO for {varname} =======")
    for i, uid in enumerate(ids):
        ax = axes[i]

        # get df for each user ID
        # map date to value"
        user_df = valence_df[valence_df['id'] == uid].set_index('date')['value']
        # if two values on the same datetime, take the mean:
        user_daily = user_df.groupby(level=0).mean()
        # create complete calendar, including missing dates
        full_range = pd.date_range(user_daily.index.min(), user_daily.index.max(), freq='D')
        # give nan-values to missing dates
        user_full = user_daily.reindex(full_range)

        #print missing values
        n_total = len(user_full)
        n_missing = user_full.isna().sum()
        frac = n_missing / n_total
        print(f"- {uid}: {n_missing} missing / {n_total} total ({frac:.1%})")

        # plot existing data points
        ax.plot(user_full.index, user_full.values, marker='o', linestyle='-', color='steelblue', linewidth=1.2, markersize=3)

        # plot missing data
        in_gap = False
        gap_start_val = None
        gap_start_date = None

        for date, val in user_full.items():
            if pd.isna(val):
                if not in_gap:
                    before = user_full[:date].dropna()
                    if not before.empty:
                        gap_start_date = before.index[-1]
                        gap_start_val = before.iloc[-1]
                    in_gap = True
                ax.plot(date, 0, marker='o', color='red',
                        markersize=4, zorder=5, transform=ax.get_xaxis_transform())
            else:
                if in_gap and gap_start_date is not None:
                    ax.plot([gap_start_date, date], [gap_start_val, val],
                            color='red', linestyle='--', linewidth=1.0, zorder=4)
                in_gap = False
                gap_start_val = None
                gap_start_date = None

        ax.set_title(str(uid), fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for j in range(len(ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{varname} per user — missing values in red', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f'Figures/full_calendar_all_ids_{varname}.png')
    plt.show()

def plot_valence_per_user_with_missing(df, varname: str = 'circumplex.valence'):
    # all rows for this variable, including those with NaN value
    valence_df = df[df['variable'] == varname].copy()
    valence_df = valence_df.sort_values(['id', 'date'])
    ids = valence_df['id'].unique()

    ncols = 6
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3), sharey=True)
    axes = axes.flatten()

    print(f"==== MISSING VALUE INFO for {varname} =======")
    for i, uid in enumerate(ids):
        ax = axes[i]

        user_df = valence_df[valence_df['id'] == uid].copy()

        user_series = user_df.set_index('date')['value']

        present = user_series[user_series.notna()]   # rows with actual values
        missing = user_series[user_series.isna()]    # rows where date exists but value is nan

        n_total = len(user_series)
        n_missing = len(missing)
        print(f"- {uid}: {n_missing} missing / {n_total} total ({n_missing/n_total:.1%})")

        # plot existing values
        ax.plot(present.index, present.values, marker='o', linestyle='-',
                color='steelblue', linewidth=1.2, markersize=3)

        # mark missing rows as red dots on x-axis
        for md in missing.index:
            ax.plot(md, 0, marker='o', color='red',
                    markersize=4, zorder=5, transform=ax.get_xaxis_transform())

        ax.set_title(str(uid), fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(len(ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{varname} per user — missing values in red', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f'Figures/missinvals_all_ids_{varname}.png')
    plt.show()

plot_valence_per_user_full_calendar(df)
plot_valence_per_user_full_calendar(df, varname="circumplex.arousal")
plot_valence_per_user_with_missing(df)
plot_valence_per_user_with_missing(df, varname="circumplex.arousal")



# TODO
# histogram of mood including mean, median, min and max value
# histogram of date including mean, median, min and max value
# print number of distinct ids
# print number of days
# find any missing values
print("\n\n")
plot_mood_vs_screentime(df)
plot_valence_and_arousal_vs_screentime(df)

histogram_of_mood(df)
print("\n\n")
histogram_dates(df)

cleaned_df = missing_imputation(df, 'circumplex.valence', linear_interpol)
cleaned_df = missing_imputation(cleaned_df, 'circumplex.arousal', linear_interpol)
# still one value missing in circumplex.valence; try backward interpol
cleaned_df = missing_imputation(cleaned_df, 'circumplex.valence', backward_interpol)
print(cleaned_df.head(20))
print_general_info(cleaned_df)