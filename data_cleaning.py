#
#
#


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_mood_smartphone.csv')

print(df.head(20))


# plot average mood vs average screen time for each [id]
def plot_mood_vs_screentime(df):
    # Create pivot table of retrieve columns 'screen' and 'mood'
    # also takes average value of for each id over all dates
    avg_df = df.pivot_table(index='id', columns='variable', values='value', aggfunc='mean').reset_index()
    print(avg_df.head(20))

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

#  make datetime
df['date'] = pd.to_datetime(df['time'], errors='coerce')

# print all distict options in 'variable'
distinct_variables = df['variable'].unique().tolist()
print(distinct_variables)

plot_mood_vs_screentime(df)
plot_valence_and_arousal_vs_screentime(df)

histogram_of_mood(df)
print("\n\n")
histogram_dates(df)


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

# histogram of mood including mean, median, min and max value
# histogram of date including mean, median, min and max value
# print number of distinct ids
# print number of days
# find any missing values