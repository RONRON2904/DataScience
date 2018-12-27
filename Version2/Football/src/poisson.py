import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

fields = ['month',
          'pHomeWin1',
          'pDraw1',
          'pAwayWin1',
          'pHomeWin2',
          'pDraw2',
          'pAwayWin2',
          'pHomeWin3',
          'pDraw3',
          'pAwayWin3',
          'homePoints',
          'awayPoints',
          'homeRank',
          'awayRank',
          'homeGlobalRank',
          'awayGlobalRank',
          'homeAccuracy',
          'awayAccuracy',
          'homeEfficiency', 
          'awayEfficiency',        
          'homeVulnerability',
          'awayVulnerability',
          'oddHome',
          'oddDraw',
          'oddAway', 
          'res']

X = np.load('../data/NPY_FILES/V2/X.npy')
y = np.load('../data/NPY_FILES/V2/y.npy')
X = np.column_stack((X, y))
df = pd.DataFrame(X, columns=fields)
print(df.res.value_counts()/df.shape[0])
months = df.month.unique()
print(months)

def pred(x):
    if max(x) == x['pHomeWin1']:
        return 1
    elif max(x) == x['pDraw1']:
        return 2
    return 3

for month in months:
    df_month = df.loc[df['month'] == month]
    df_poisson = df_month[['pHomeWin1', 'pDraw1', 'pAwayWin1']]
    y = df_month['res'].values.reshape(-1, 1)
    y_pred = df_poisson.apply(lambda x: pred(x), axis=1).values.reshape(-1, 1)
    print(accuracy_score(y_pred, y))

