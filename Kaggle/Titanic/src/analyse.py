import pandas as pd
import matplotlib.pyplot as plt

"""
This stat has been computed in order to see the correlation between a known cabin
for a person and the fact he survived or not.

The purpose afterwards is to put a binary information about it into the X_train.npy file:
1 if the cabin is known else 0

Indeed, in the given csv files we can't put this information into a numpy array as it is a string or NaN
"""
def cabin_stats():
    df = pd.read_csv('data/CSV_FILES/train.csv')
    survived = df.loc[df['Survived'] == 1]
    dead = df.loc[df['Survived'] == 0]
    survivedStr = 0
    deadStr = 0

    for i in range(survived.shape[0]):
        line = survived.iloc[i]
        if isinstance(line['Cabin'], str):
            survivedStr += 1

    for i in range(dead.shape[0]):
        line = dead.iloc[i]
        if isinstance(line['Cabin'], str):
            deadStr += 1

    return float(survivedStr)/survived.shape[0], float(deadStr)/dead.shape[0]

def ticket_stats():
    df = pd.read_csv('data/CSV_FILES/train.csv')
    survived = df.loc[df['Survived'] == 1]
    dead = df.loc[df['Survived'] == 0]

    for i in range(survived.shape[0]):
        line = survived.iloc[i]
        print('Survived', line['Ticket'])

    for i in range(dead.shape[0]):
        line = dead.iloc[i]
        print('Dead', line['Ticket'])

"""
Is there any correlation between a Mr. male / not Mr. male and the fact that he survived or not?
Same for female with 'Mrs.' or not 'Mrs.' status ?
This stats aims at showing whether it is worth taking this information into account or not.
"""
def name_stats():
    df = pd.read_csv('data/CSV_FILES/train.csv')
    survived_male = df.loc[(df['Survived'] == 1) & (df['Sex'] == 'male')]
    survived_female = df.loc[(df['Survived'] == 1) & (df['Sex'] == 'female')]
    dead_male = df.loc[(df['Survived'] == 0) & (df['Sex'] == 'male')]
    dead_female = df.loc[(df['Survived'] == 0) & (df['Sex'] == 'female')]

    mr_survived = 0
    not_mr_survived = 0
    mr_dead = 0
    not_mr_dead = 0

    mrs_survived = 0
    not_mrs_survived = 0
    mrs_dead = 0
    not_mrs_dead = 0


    for i in range(survived_male.shape[0]):
        line = survived_male.iloc[i]
        if 'Mr.' in line['Name']:
            mr_survived += 1
        else :
            not_mr_survived += 1

    for i in range(dead_male.shape[0]):
        line = dead_male.iloc[i]
        if 'Mr.' in line['Name']:
            mr_dead += 1
        else :
            not_mr_dead += 1

    for i in range(survived_female.shape[0]):
        line = survived_female.iloc[i]
        if 'Mrs.' in line['Name']:
            mrs_survived += 1
        else :
            not_mrs_survived += 1

    for i in range(dead_female.shape[0]):
        line = dead_male.iloc[i]
        if 'Mr.' in line['Name']:
            mrs_dead += 1
        else :
            not_mrs_dead += 1

    total_mr = mr_survived + mr_dead
    total_not_mr = not_mr_survived + not_mr_dead
    total_mrs = mrs_survived + mrs_dead
    total_not_mrs = not_mrs_survived + not_mrs_dead

    return mr_survived/total_mr, mr_dead/total_mr, not_mr_survived/total_not_mr, not_mr_dead/total_not_mr, mrs_survived/total_mrs, mrs_dead/total_mrs, not_mrs_survived/total_not_mrs, not_mrs_dead/total_not_mrs

"""
This Function plots the bar histogram to show how correlated is a feature with the target Survived  
"""
def plot_bar(feature):
    df = pd.read_csv('data/CSV_FILES/train.csv')
    survived = df.loc[df['Survived'] == 1][feature].value_counts()
    dead = df.loc[df['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()

#plot_bar('SibSp')
