import numpy as np
from xgboost import XGBRegressor as XGBR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, minmax_scale, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, normalize
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

def kelly(odd, p):
    return (p*(odd - 1) - (1 - p))/(odd - 1)
    
def custom_scorer(y_true, Y_pred):
    score = 200.
    cpt = 0
    x = []
    y = []
    value=0.95
    cpt = 0
    tot = 0
    bk = 200.
    mise = 20.
    nb_win = 0
    nb_loose = 0
    for i in range(Y_pred.shape[0]):
        p_win = Y_pred[i, 0]
        p_draw = Y_pred[i, 1]
        p_loose = Y_pred[i, 2]
        if p_win > 0.:
            if 1/p_win < X[i, -3] and 1/p_win > value*X[i, -3]:
                ratio = kelly(X[i, -3], p_win)
                mise = ratio*score
                cpt += 1
                tot += mise
                if y_true[i] == 1:
                    score += mise*(X[i, -3] - 1)
                    nb_win += 1
                    print('Win WIN odd:', X[i, -3], 'bk:', score, 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    print('Loose WIN odd:', X[i, -3], 'bk:',score, 'mise:', mise)
                x.append(cpt)
                y.append(score)
        if p_draw > 0.:
            if 1/p_draw < X[i, -2] and 1/p_draw > value*X[i, -2]:
                ratio = kelly(X[i, -2], p_draw)
                mise = ratio*score
                cpt += 1
                tot += mise
                if y_true[i] == 2:
                    score += mise*(X[i, -2] - 1)
                    nb_win += 1
                    print('Win DRAW odd:', X[i, -2], 'bk:',score, 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    print('Loose DRAW odd:', X[i, -2], 'bk:',score, 'mise:', mise)
                x.append(cpt)
                y.append(score)
        if p_loose > 0.:
            if 1/p_loose < X[i, -1] and 1/p_loose > value*X[i, -1]:
                ratio = kelly(X[i, -1], p_loose)
                mise = ratio*score
                cpt += 1
                tot += mise
                if y_true[i] == 3:
                    score += mise*(X[i, -1] - 1)
                    nb_win += 1
                    print('Win LOOSE odd:', X[i, -1], 'bk:',score, 'mise:', mise)
                else:
                    score -= mise
                    nb_loose += 1
                    print('Loose LOOSE odd:', X[i, -1], 'bk:',score, 'mise:', mise)
                x.append(cpt)
                y.append(score)
    plt.plot(x, y)
    plt.show()
    print(score, tot, cpt, nb_win, nb_loose)
    return 100*(score-bk)/tot


def custom_scorer2(y_true, Y_pred):
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
    for i in range(Y_pred.shape[0]):
        if Y_pred[i] == 1:
            ratio = kelly(X[i, -3], 2.)
            mise = ratio*score
            cpt += 1
            tot += mise
            if y_true[i] == 1:
                score += mise*(X[i, -3] - 1)
                nb_win += 1
                print('Win WIN odd:', X[i, -3], 'bk:', score, 'mise:', mise)
            else:
                score -= mise
                nb_loose += 1
                print('Loose WIN odd:', X[i, -3], 'bk:',score, 'mise:', mise)
            x.append(cpt)
            y.append(score)
        if Y_pred[i] == 2:
            ratio = kelly(X[i, -2], 2.)
            mise = ratio*score
            cpt += 1
            tot += mise
            if y_true[i] == 2:
                score += mise*(X[i, -2] - 1)
                nb_win += 1
                print('Win DRAW odd:', X[i, -2], 'bk:',score, 'mise:', mise)
            else:
                score -= mise
                nb_loose += 1
                print('Loose DRAW odd:', X[i, -2], 'bk:',score, 'mise:', mise)
            x.append(cpt)
            y.append(score)
        if Y_pred[i] == 3:
            ratio = kelly(X[i, -1], 2.)
            mise = ratio*score
            cpt += 1
            tot += mise
            if y_true[i] == 3:
                score += mise*(X[i, -1] - 1)
                nb_win += 1
                print('Win LOOSE odd:', X[i, -1], 'bk:',score, 'mise:', mise)
            else:
                score -= mise
                nb_loose += 1
                print('Loose LOOSE odd:', X[i, -1], 'bk:',score, 'mise:', mise)
            x.append(cpt)
            y.append(score)
    plt.plot(x, y)
    plt.show()
    print(score, tot, cpt, nb_win, nb_loose)
    return 100*(score-bk)/tot


def book_score(X, y):
    cpt = 0
    for i in range(X.shape[0]):
        if (X[i, -3] == np.min(X[i, -3])) and (y[i] == 1):
            cpt += 1
        elif (X[i, -2] == np.min(X[i, -3])) and (y[i] == 2):
            cpt += 1
        elif (X[i, -1] == np.min(X[i, -3])) and (y[i] == 3):
            cpt += 1
    return float(cpt/y.shape[0])



X = np.load('../data/NPY_FILES/V2/X_ligue1.npy')
y = np.load('../data/NPY_FILES/V2/y_ligue1.npy') 




fields = ['month',
          'p_home_win1',
          'p_draw1',
          'p_away_win1',
          'p_home_win2',
          'p_draw2',
          'p_away_win2',
          'p_home_win3',
          'p_draw3',
          'p_away_win3',
          'd_points',
          'd_rank',
          'd_global_rank',
          'd_accuracy',
          'd_efficiency',
          'd_vulnerability',
          'oddHome',
          'oddDraw',
          'oddAway']

def heatMap(df):
    corr = df.corr()
    #invCorr = np.linalg.inv(corr.values)
    #corr = pd.DataFrame(invCorr, columns= fields).corr()
    #Plot figsize
    fig, ax = plt.subplots(figsize=(10, 10))
    #Generate Color Map, red & blue
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns)
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    plt.show()

#df = pd.DataFrame(X, columns=fields)
#heatMap(df)

def decorrelate(X):
    df = pd.DataFrame(X)
    corr = df.corr()
    invCorr = np.linalg.inv(corr.values)
    X_decorr = np.array([])
    for i in range(corr.shape[1]):
        X_decorr = np.append(X_decorr, np.dot(X, invCorr[:, i].reshape(-1, 1)))
    return X_decorr.reshape(X.shape)

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

#X_copy = X
#X_ = PCA().fit_transform(X) #decorrelate(X)

#print(X_.shape)
#df = pd.DataFrame(X_, columns=fields)
#heatMap(df)


_, X_ctest, _, _ = train_test_split(X, y, test_size=0.16, shuffle=False)
#X_train = X_train

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

n=4
X = distributions[n][1]
X_ = whiten(X, 'cholesky')
#df = pd.DataFrame(X_, columns=fields)
#heatMap(df)
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.16, shuffle=False)

#xParam = {"n_estimators": (100, 200, 500), "learning_rate": (0.1, 0.01), "max_depth": (1, 2)}
#x = GridSearchCV(XGBC(), xParam, cv=3)
x = XGBR(n_estimators=2000, learning_rate=0.01, max_depth=5)
x.fit(X_train, y_train[:, [0]].reshape(-1, 1))
y_pred = x.predict(X_test).reshape(-1, 1)
y = np.column_stack((y_pred, y_test[:, [0]])).reshape(-1, 2)
print(y[:30])
print(mse(y_pred, y_test[:, [0]]))
"""
x = RandomForestClassifier(n_estimators=1000, max_depth=2)
x.fit(train_distributions[n][1], y_train)

x = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
x.fit(train_distributions[n][1], y_train)

x = GaussianNB()
x.fit(train_distributions[n][1], y_train)

sgd = SGDClassifier()
sgd.fit(train_distributions[n][1], y_train)

#print('XGBC: ', x.score(X_test, y_test))
#print(book_score(X_ctest, y_test))
#X = X_ctest
#print(custom_scorer(y_test, x.predict_proba(X_test)))
#print(x.predict_proba(X_test[-20:]))

print('ADBC: ', adbc.score(test_distributions[n][1], y_test))
print('RFC: ', rfc.score(test_distributions[n][1], y_test))

print(gnb.score(test_distributions[n][1], y_test))
print(sgd.score(test_distributions[n][1], y_test))

y_pred = x.predict_proba(test_distributions[n][1])
#print(y_pred)
#print(x.feature_importances_)
X = X_ctest
print(custom_scorer(y_test, y_pred))
"""