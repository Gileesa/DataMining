#
#
#


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_mood_smartphone.csv')

print(df.head(20))

# print all distict options in 'variable'
distinct_variables = df['variable'].unique().tolist()
print(distinct_variables)


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


plot_mood_vs_screentime(df)
plot_valence_and_arousal_vs_screentime(df)

# histogram of mood
# number of distinct ids
# number of days
# reduce DF