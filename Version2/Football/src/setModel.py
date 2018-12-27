import numpy as np
from xgboost import XGBRegressor as XGBR
from sklearn.preprocessing import MinMaxScaler, minmax_scale, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, normalize
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

def custom_scorer(y_true, y_pred, odds):
    score = 200.
    cpt = 0
    x = []
    y = []
    cpt = 0
    tot = 0
    bk = 200.
    mise = 20.
    value = 0.98
    nb_win = 0
    nb_loose = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] > 0.:
            if 1/y_pred[i] < odds[i] and 1/y_pred[i] > value*odds[i]:
                #ratio = kelly(X[i, -3], p_win)
                #mise = ratio*score
                cpt += 1
                tot += mise
                if y_true[i] == 2:
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

X = np.load('../data/NPY_FILES/V2/X_ligue1.npy')
y = np.load('../data/NPY_FILES/V2/y_ligue1.npy') 
odds = X[:, [-3, -2, -1]]
X = X[:, :-3]
#y_t = np.load('../data/NPY_FILES/y_ligue1.npy')

#_, X_ctest, _, _ = train_test_split(X, y, test_size=0.16, shuffle=False)
#Odds = X_ctest[:, [-3, -2, -1]]

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

n=0
X = distributions[n][1]
X_ = X #whiten(X, 'cholesky')
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.16, shuffle=False)
#_, _, _, y_true = train_test_split(X_, y_t, test_size=0.16, shuffle=False)
x = XGBR(n_estimators=1000, learning_rate=0.01, max_depth=3, reg_lambda=0.15)
x.fit(X_train, y_train[:, [0]].reshape(-1, 1))
y_pred = x.predict(X_test).reshape(-1, 1)
print(np.column_stack((y_pred, y_test[:, [0]]))[:100])
print(np.sqrt(mse(y_pred, y_test[:, [0]]))) 
print(x.feature_importances_) 
#print(custom_scorer(y_true, y_pred, Odds[:, [1]]))

