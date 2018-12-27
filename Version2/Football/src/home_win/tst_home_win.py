import numpy as np
from xgboost import XGBRegressor as XGBC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

def kelly(odd, p):
    return 0.03
    #return (p*(odd - 1) - (1 - p))/(odd - 1)

def custom_scorer(y_true, y_pred, odds):
    score = 200.
    cpt = 0
    x = []
    y = []
    cpt = 0
    tot = 0
    bk = 200.
    mise = 20.
    nb_win = 0
    nb_loose = 0
    for i in range(y_pred.shape[0]):
       # print(odds[i])
        if y_pred[i] > 0.:
            if ((1/y_pred[i] < odds[i]) & (odds[i] > 4.)) | (y_pred[i] > 0.71):
            #if (y_pred[i] > 0.7):
                #ratio = kelly(odds[i], y_pred[i])
                #mise = ratio*score
                cpt += 1
                tot += mise
                if y_true[i] == 1:
                    score += mise*(odds[i] - 1)
                    nb_win += 1
                    print('Win odd:', odds[i], 'bk:', score, 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    print('Loose odd:', odds[i], 'bk:',score, 'mise:', mise)
                x.append(cpt)
                y.append(float(score))
    plt.plot(x, y)
    plt.show()
    print(score, tot, cpt, nb_win, nb_loose)
    return 100*(score-bk)/tot

def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None
    
    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), V_sqrt)
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), V_sqrt)
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)

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
          'oddH',
          'oddD',
          'oddA',
          'res',
          'oddHome',
          'oddDraw',
          'oddAway']

X_train = np.load('../../data/NPY_FILES/V2/X_train.npy')
y_train = np.load('../../data/NPY_FILES/V2/y_train.npy')
X_test = np.load('../../data/NPY_FILES/V2/X_test.npy')
y_test = np.load('../../data/NPY_FILES/V2/y_test.npy')
odds_train = np.load('../../data/NPY_FILES/V2/odds_train.npy')
odds_test = np.load('../../data/NPY_FILES/V2/odds_test.npy')

def func(x):
    if x < 0.5:
        return 0
    return 1

vfunc = np.vectorize(func)

df_train = pd.DataFrame(np.column_stack((X_train, y_train, odds_train)), columns=fields)
df_train = df_train[(df_train['month'] != 8) & (df_train['month'] != 9)]
y_train = df_train.apply(lambda x: 1 if (x['res'] == 1) else 0, axis=1).values.reshape(-1, 1)
odds_train = df_train['oddH'].values.reshape(-1, 1)
df_train = df_train.drop(columns=['res', 'oddHome', 'oddDraw', 'oddAway'])
X_train = df_train.values
X_train_whiten = whiten(X_train)


df_test = pd.DataFrame(np.column_stack((X_test, y_test, odds_test)), columns=fields)
df_test = df_test[(df_test['month'] != 8) & (df_test['month'] != 9)]
y_test = df_test.apply(lambda x: 1 if (x['res'] == 1) else 0, axis=1).values.reshape(-1, 1)
odds_test = df_test['oddH'].values.reshape(-1, 1)
df_test = df_test.drop(columns=['res', 'oddHome', 'oddDraw', 'oddAway'])
X_test = df_test.values
X_test_whiten = whiten(X_test)

#xgbc = XGBC(n_estimators=2000, learning_rate=0.001, objective='binary:logistic')
xgbc = XGBC(n_estimators=2000, learning_rate=0.001, max_depth=8, objective='binary:logistic')
xw = xgbc.fit(X_train, y_train)
y_pred = xw.predict(X_test)
#print(np.column_stack((y_pred[200:500], y_test[200:500])))
#print(accuracy_score(y_pred, y_test))
#print(x.score(X_test, y_test), xw.score(X_wtest, y_test))
print(custom_scorer(y_test, y_pred, odds_test))
