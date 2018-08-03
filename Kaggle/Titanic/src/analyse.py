import pandas as pd

"""
This stat has been computed in order to see the correlation between a known cabin 
for a person and the fact he survived or not.

The purpose afterwards is to put a binary information about it into the X_train.npy file:
1 if the cabin is known else 0

Indeed, in the given csv files we can't put this information into a numpy array as it is a string or NaN   
"""
def cabin_stats():
    df = pd.read_csv('data/train.csv')
    survived = df.loc[df['Survived'] == 1]
    dead = df.loc[df['Survived'] == 0]
    survivedStr = 0
    deadStr = 0
    
    for i in range(survived.shape[0]):
        line = survived.iloc[i]
        if isinstance(line['Cabin'], basestring):
            survivedStr += 1
            
    for i in range(dead.shape[0]):
        line = dead.iloc[i]
        if isinstance(line['Cabin'], basestring):
            deadStr += 1

    return float(survivedStr)/survived.shape[0], float(deadStr)/dead.shape[0]

