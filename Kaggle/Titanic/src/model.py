from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from data_preprocessing import to_npy
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.preprocessing import normalize, MinMaxScaler, minmax_scale, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer

import numpy as np
import pickle

X_train = np.load('data/NPY_FILES/Xm_train.npy')[:, [0, 1, 4, 5, 6, 7]]
y_train = np.load('data/NPY_FILES/ym_train.npy')
X_val = np.load('data/NPY_FILES/Xm_val.npy')[:, [0, 1, 4, 5, 6, 7]]
y_val = np.load('data/NPY_FILES/ym_val.npy')

#Try some standardisation/normalization on the dataset
train_distributions = [X_train,
                       StandardScaler().fit_transform(X_train),
                       MinMaxScaler().fit_transform(X_train),
                       MaxAbsScaler().fit_transform(X_train),
                       RobustScaler(quantile_range=(25, 75)).fit_transform(X_train),
                       QuantileTransformer(output_distribution='uniform').fit_transform(X_train),
                       QuantileTransformer(output_distribution='normal').fit_transform(X_train),
                       Normalizer().fit_transform(X_train)]

val_distributions = [X_val,
                     StandardScaler().fit_transform(X_val),
                     MinMaxScaler().fit_transform(X_val),
                     MaxAbsScaler().fit_transform(X_val),
                     RobustScaler(quantile_range=(25, 75)).fit_transform(X_val),
                     QuantileTransformer(output_distribution='uniform').fit_transform(X_val),
                     QuantileTransformer(output_distribution='normal').fit_transform(X_val),
                     Normalizer().fit_transform(X_val)]

n=0

# Test various classifiers 

lgr = LogisticRegression()
lgr.fit(train_distributions[n], y_train)
pickle.dump(lgr, open('data/Models/lgr_male.pkl', 'wb'))

knc = KNeighborsClassifier(n_neighbors=10, weights='distance')
knc.fit(train_distributions[n], y_train)

dtc = DecisionTreeClassifier()
dtc.fit(train_distributions[n], y_train)

rf = RandomForestClassifier(n_estimators=400)
rf.fit(train_distributions[n], y_train)

adb = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
adb.fit(train_distributions[n], y_train)

mlp = MLPClassifier(hidden_layer_sizes=100, activation='logistic', learning_rate='adaptive')
mlp.fit(train_distributions[n], y_train)

svc = SVC()
svc.fit(train_distributions[n], y_train)


lsvc = LinearSVC(C=0.1)
lsvc.fit(train_distributions[n], y_train)

gnb = GaussianNB()
gnb.fit(train_distributions[n], y_train)

qda = QuadraticDiscriminantAnalysis()
qda.fit(train_distributions[n], y_train)

xgbc = XGBClassifier(n_estimators=110, learning_rate=0.08, max_depth=6)
xgbc.fit(train_distributions[n], y_train)
#pickle.dump(lgr, open('data/Models/xgb_female.pkl', 'wb'))

print('Score LogisticRegression :', lgr.score(val_distributions[n], y_val))
print('Score DecisionTreeClassifier :', dtc.score(val_distributions[n], y_val))
print('Score KNeighborsClassifier :', knc.score(val_distributions[n], y_val))

print('Score RandomForestClassifier :', rf.score(val_distributions[n], y_val))


print('Score AdaBoostClassifier :', adb.score(val_distributions[n], y_val))
print('Score MLPClassifier :', mlp.score(val_distributions[n], y_val))
print('Score SVC :', svc.score(val_distributions[n], y_val))
print('Score LinearSVC :', lsvc.score(val_distributions[n], y_val))
print('Score GNB :', gnb.score(val_distributions[n], y_val))
print('Score QuadraticDiscriminantAnalysis :', qda.score(val_distributions[n], y_val))
print('Score XGBClassifier :', xgbc.score(val_distributions[n], y_val))
