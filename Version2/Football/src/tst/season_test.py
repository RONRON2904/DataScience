import numpy as np
from xgboost import XGBRegressor as XGBR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def custom_scorer(y_true, y_hpred, y_apred, odds_h, odds_a, rows):
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
    print(y_hpred.shape, y_apred.shape, y_true.shape, rows)
    for i in range(y_true.shape[0]):
        mise = min(max(0.05*score, 5), 5000)
        if i < rows:
            if (((1/y_hpred[i] < odds_h[i]) & (odds_h[i] > 5.)) | (y_hpred[i] > 0.7)):
                cpt += 1
                tot += mise
                if y_true[i] == 1:
                    score += mise*(odds_h[i] - 1)
                    nb_win += 1
                    print('Win Home:', odds_h[i], 'bk:', score, 'proba:', y_hpred[i], 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    print('Loose Home:', odds_h[i], 'bk:', score, 'proba:', y_hpred[i], 'mise:', mise)
                x.append(cpt)
                y.append(float(score))
            if (y_apred[i] > 0.62):
                cpt += 1
                tot += mise
                if y_true[i] == 3:
                    score += mise*(odds_a[i] - 1)
                    nb_win += 1
                    print('Win Away:', odds_a[i], 'bk:', score, 'proba:', y_apred[i], 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    print('Loose Away:', odds_a[i], 'bk:',score, 'proba:', y_apred[i], 'mise:', mise)
                x.append(cpt)
                y.append(float(score))
        if i >= rows:
            if (((1/y_hpred[i] < odds_h[i]) & (odds_h[i] > 5.)) | (y_hpred[i] > 0.6)):
                cpt += 1
                tot += mise
                if y_true[i] == 1:
                    score += mise*(odds_h[i] - 1)
                    nb_win += 1
                    print('Win Home:', odds_h[i], 'bk:', score, 'proba:', y_hpred[i], 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    print('Loose Home:', odds_h[i], 'bk:', score, 'proba:', y_hpred[i], 'mise:', mise)
                x.append(cpt)
                y.append(float(score))
            if (y_apred[i] > 0.5):
                cpt += 1
                tot += mise
                if y_true[i] == 3:
                    score += mise*(odds_a[i] - 1)
                    nb_win += 1
                    print('Win Away:', odds_a[i], 'bk:', score, 'proba:', y_apred[i], 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    print('Loose Away:', odds_a[i], 'bk:',score, 'proba:', y_apred[i], 'mise:', mise)
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
          'res']

X_train = np.load('../../data/NPY_FILES/V2/X_train.npy')
y_train = np.load('../../data/NPY_FILES/V2/y_train.npy')
X_test = np.load('../../data/NPY_FILES/V2/X_test.npy')
y_test = np.load('../../data/NPY_FILES/V2/y_test.npy')

df_train = pd.DataFrame(np.column_stack((X_train, y_train)), columns=fields)
df_train_first = df_train[(df_train['month'] == 10) | (df_train['month'] == 11) | (df_train['month'] == 12) | (df_train['month'] == 1)] 
df_train_second = df_train[(df_train['month'] == 2) | (df_train['month'] == 3) | (df_train['month'] == 4) | (df_train['month'] == 5)] 


df_test = pd.DataFrame(np.column_stack((X_test, y_test)), columns=fields)
df_test_first = df_test[(df_test['month'] == 10) | (df_test['month'] == 11) | (df_test['month'] == 12) | (df_test['month'] == 1)] 
df_test_second = df_test[(df_test['month'] == 2) | (df_test['month'] == 3) | (df_test['month'] == 4) | (df_test['month'] == 5)] 
odds_h_first = df_test_first['oddH'].values.reshape(-1, 1)
odds_h_second = df_test_second['oddH'].values.reshape(-1, 1)
odds_a_first = df_test_first['oddA'].values.reshape(-1, 1)
odds_a_second = df_test_second['oddA'].values.reshape(-1, 1)

odds_h = np.concatenate((odds_h_first, odds_h_second)).reshape(-1, 1)
odds_a = np.concatenate((odds_a_first, odds_a_second)).reshape(-1, 1)

y_test_first = df_test_first['res'].values.reshape(-1, 1)
y_test_second = df_test_second['res'].values.reshape(-1, 1)
y_test = np.concatenate((y_test_first, y_test_second)).reshape(-1, 1)

y_home_first = df_train_first.apply(lambda x: 1 if (x['res'] == 1) else 0, axis=1).values.reshape(-1, 1)
y_home_second = df_train_second.apply(lambda x: 1 if (x['res'] == 1) else 0, axis=1).values.reshape(-1, 1)
y_away_first = df_train_first.apply(lambda x: 1 if (x['res'] == 3) else 0, axis=1).values.reshape(-1, 1)
y_away_second = df_train_second.apply(lambda x: 1 if (x['res'] == 3) else 0, axis=1).values.reshape(-1, 1)


df_train_first = df_train_first.drop(columns=['res'])
df_train_second = df_train_second.drop(columns=['res'])

df_test_first = df_test_first.drop(columns=['res'])
df_test_second = df_test_second.drop(columns=['res'])

X_train_first = df_train_first.values
X_train_second = df_train_second.values
X_train_first = whiten(X_train_first)
X_train_second = whiten(X_train_second)


X_test_first = df_test_first.values
X_test_second = df_test_second.values
X_test_first = whiten(X_test_first)
X_test_second = whiten(X_test_second)

xgb_home_first = XGBR(n_estimators=2000, learning_rate=0.001, objective='binary:logistic')
xgb_home_second = XGBR(n_estimators=5000, learning_rate=0.001, max_depth=10, objective='binary:logistic')
xgb_away_first = XGBR(n_estimators=2000, learning_rate=0.001, max_depth=4, objective='binary:logistic')
xgb_away_second = XGBR(n_estimators=3000, learning_rate=0.001, max_depth=4, objective='binary:logistic')

"""
X_train_first, _, y_home_first, _, y_away_first, _ = train_test_split(X_train_first, y_home_first, y_away_first, train_size=0.7)
X_train_second, _, y_home_second, _, y_away_second, _ = train_test_split(X_train_second, y_home_second, y_away_second, train_size=0.7)
"""
xgb_hf = xgb_home_first.fit(X_train_first, y_home_first)
xgb_hs = xgb_home_second.fit(X_train_second, y_home_second)
xgb_af = xgb_away_first.fit(X_train_first, y_away_first)
xgb_as = xgb_away_second.fit(X_train_second, y_away_second)

hf_pred = xgb_hf.predict(X_test_first)
hs_pred = xgb_hs.predict(X_test_second)
af_pred = xgb_af.predict(X_test_first)
as_pred = xgb_as.predict(X_test_second)

h_pred = np.concatenate((hf_pred, hs_pred)).reshape(-1, 1)
a_pred = np.concatenate((af_pred, as_pred)).reshape(-1, 1)

rows = X_test_first.shape[0]
custom_scorer(y_test, h_pred, a_pred, odds_h, odds_a, rows)