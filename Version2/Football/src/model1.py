import numpy as np
from xgboost import XGBClassifier as XGBC
from sklearn.preprocessing import MinMaxScaler, minmax_scale, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, normalize
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def custom_scorer(y_pred, y_true, odds, cat, plot=False):
    score = 200.
    cpt = 0
    x = []
    y = []
    cpt = 0
    tot = 0
    bk = 200.
    mise = 20.
    max_value = 0.999
    min_value = 0.1
    nb_win = 0
    nb_loose = 0
    diff = np.array([])
    bet = np.array([])
    for i in range(y_pred.shape[0]):
        p_win = y_pred[i][0]
        #new_features = np.append(new_features, odds[i] - (1/p_win))
        if p_win != 0.:
            if 1/p_win < max_value * odds[i][0] and 1/p_win > min_value * odds[i][0]:
                cpt += 1
                tot += mise
                if y_true[i] == cat:
                    score += mise*(odds[i] - 1)
                    nb_win += 1
                    diff = np.append(diff, odds[i] - (1/p_win))
                    bet = np.append(bet, 3)
                    print('Win odd:', odds[i], 'bk:', score, 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    diff = np.append(diff, odds[i] - (1/p_win))
                    bet = np.append(bet, 1)
                    print('Loose odd:', odds[i], 'bk:',score, 'mise:', mise)
                x.append(cpt)
                y.append(float(score))
            else:
                diff = np.append(diff, odds[i] - (1/p_win))
                bet = np.append(bet, 2)
    if plot:
        plt.plot(x, y)
        plt.show()
    roi = 100*(score-bk)/tot
    print(score, tot, cpt, nb_win, nb_loose, roi)
    return diff.reshape(-1, 1), bet.reshape(-1, 1)

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

X_ = np.load('../data/NPY_FILES/X.npy')
y_ = np.load('../data/NPY_FILES/V2/y.npy')
odds_ = np.load('../data/NPY_FILES/V2/odds.npy')
s = 0
for i in range(10):
    X, y, odds = shuffle(X_, y_, odds_)
    X = X[:, :-3]
    #X = whiten(X, 'cholesky')
    """
    selector = SelectPercentile(f_classif, percentile=40)
    selector.fit(X, y)
    X = selector.transform(X)
    """
    distributions = [
        ('Unscaled data', X),
        ('Data after standard scaling', StandardScaler().fit_transform(X)),
        ('Data after min-max scaling', MinMaxScaler().fit_transform(X)),
        ('Data after max-abs scaling', MaxAbsScaler().fit_transform(X)),
        ('Data after robust scaling', RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
        ('Data after quantile transformation (uniform pdf)', QuantileTransformer(output_distribution='uniform').fit_transform(X)),
        ('Data after quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal').fit_transform(X)),
        ('Data after sample-wise L2 normalizing', Normalizer().fit_transform(X))
    ]
    n = 0
    X = distributions[n][1]

    category = 1
    odd_cat = category - 1

    X_train = X[:3445] #10340 raws => take the first third
    y_train = y[:3445]
    X_test = X[3445:] #test on the last two thirds
    y_test = y[3445:]
    y = y_test
    #odds_test = X_test[:, [-3, -2, -1]].reshape(-1, 3)
    odds_test = odds[3445:]
    x = XGBC(n_estimators=200, learning_rate=0.01)
    x.fit(X_train, y_train)
    s += x.score(X_test, y_test)
print(s/10.)
    
"""
y_pred = x.predict_proba(X_test)[:, [odd_cat]].reshape(-1, 1)

diff, bet = custom_scorer(y_pred, y_test, odds_test[:, [odd_cat]].reshape(-1, 1), category, plot=True)
y_ = bet
X_ = np.column_stack((X_test, diff))
#selector = SelectPercentile(f_classif, percentile=50)
#selector.fit(X_, y_)
#X_f = selector.transform(X_)
distributions_ = [
    ('Unscaled data', X_),
    ('Data after standard scaling', StandardScaler().fit_transform(X_)),
    ('Data after min-max scaling', MinMaxScaler().fit_transform(X_)),
    ('Data after max-abs scaling', MaxAbsScaler().fit_transform(X_)),
    ('Data after robust scaling', RobustScaler(quantile_range=(25, 75)).fit_transform(X_)),
    ('Data after quantile transformation (uniform pdf)', QuantileTransformer(output_distribution='uniform').fit_transform(X_)),
    ('Data after quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal').fit_transform(X_)),
    ('Data after sample-wise L2 normalizing', Normalizer().fit_transform(X_))
]
n = 0
X_ = distributions_[n][1]
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.4, shuffle=False)
x = XGBC(n_estimators=1000, learning_rate=0.001, max_depth=5)
x.fit(X_train, y_train)
_y_ = x.predict(X_test)
_X_ = np.column_stack((X_test, _y_))
rows = X_train.shape[0]
odds_test = odds_test[rows:]
y = y[rows:]

X_train, X_test, y_train, y_test = train_test_split(_X_, y, test_size = 0.4, shuffle=False)
rows = X_train.shape[0]
odds_test = odds_test[rows:]
x.fit(X_train, y_train)
y_pred = x.predict_proba(X_test)[:, [odd_cat]].reshape(-1, 1)
custom_scorer(y_pred, y_test, odds_test[:, [odd_cat]].reshape(-1, 1), category, plot=True)

_X_ = X_test
rows = X_train.shape[0]
_y_ = y_test[rows:]
print(_X_.shape, _y_.shape)
X_train, X_test, y_train, y_test = train_test_split(_X_, _y_, test_size=0.4, shuffle=False)
x.fit(X_train, y_train)
y_pred = x.predict_proba(X_test)[:, [0]].reshape(-1, 1)
odds = odds_test[rows:]
custom_scorer(y_pred, y_test, odds[:, [0]].reshape(-1, 1), 1, plot=True)

rows = X_train.shape[0]
odds = odds_test[rows:]
x.fit(X_train, y_train)
y_pred = x.predict_proba(X_test)[:, [0]].reshape(-1, 1)
custom_scorer(y_pred, y_test, odds[:, [0]].reshape(-1, 1), 1, plot=True)
"""