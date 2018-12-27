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

def custom_scorer(y_true, y_pred, odds, start):
    score = start
    cpt = 0
    x = []
    y = []
    cpt = 0
    tot = 0
    bk = score
    mise = 20.
    nb_win = 0
    nb_loose = 0
    for i in range(y_pred.shape[0]):
       # print(odds[i])
        if y_pred[i] > 0.:
            #if ((1/y_pred[i] < odds[i]) & (odds[i] > 20.)) | (y_pred[i] > 0.5):
            if ((y_pred[i] > 0.5)):
                ratio = kelly(odds[i], y_pred[i])
                mise = min(max(ratio*score, 5), 5000)
                cpt += 1
                tot += mise
                if y_true[i] == 1:
                    score += mise*(odds[i] - 1)
                    nb_win += 1
                   # print('Win odd:', odds[i], 'bk:', score, 'proba:', y_pred[i], 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    #print('Loose odd:', odds[i], 'bk:',score, 'proba:', y_pred[i], 'mise:', mise)
                x.append(cpt)
                y.append(float(score))
    plt.plot(x, y)
    plt.show()
    print(cpt, nb_win, nb_loose)
    return score

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

def func(x):
        if x < 0.5:
            return 0
        return 1

vfunc = np.vectorize(func)
s = 0.
w = 0
l = 0
start = 200.
for i in range(20):
    X = np.load('../../data/NPY_FILES/V2/X.npy')
    y = np.load('../../data/NPY_FILES/V2/y.npy')
    odds = np.load('../../data/NPY_FILES/V2/odds.npy')
    df = pd.DataFrame(np.column_stack((X, y, odds)), columns=fields)
    #df = df[(df['month'] == 10) | (df['month'] == 11) | (df['month'] == 12) | (df['month'] == 1)] 
    df = df[(df['month'] == 2) | (df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5)] 
    #df = df[(df['month'] != 8) & (df['month'] != 9)]
    y = df.apply(lambda x: 1 if (x['res'] == 3) else 0, axis=1).values.reshape(-1, 1)
    odds = df['oddA'].values.reshape(-1, 1)
    df = df.drop(columns=['res', 'oddHome', 'oddDraw', 'oddAway'])
    X = df.values
    X_whiten = whiten(X)
    X_wtrain, X_wtest, y_train, y_test, _, odds_test = train_test_split(X_whiten, y, odds, test_size=0.4)
    #xgbc = XGBC(n_estimators=1000, learning_rate=0.001, max_depth=4, objective='binary:logistic')
    xgbc = XGBC(n_estimators=3000, learning_rate=0.001, max_depth=6, objective='binary:logistic')
    xw = xgbc.fit(X_wtrain, y_train)
    y_pred = xw.predict(X_wtest)
    #print(np.column_stack((y_pred[200:500], y_test[200:500])))
    rows = X_wtrain.shape[0]
    #print(accuracy_score(y_pred, y_test))
    #print(x.score(X_test, y_test), xw.score(X_wtest, y_test))
    start = custom_scorer(y_test, y_pred, odds_test, start)
    print(start)
    """
    if roi > 0.:
        w += 1
    else:
        l += 1
    s += roi
print(s, s/50., w, l)
"""