import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pickle

def create_csv_file(filename, features, values):
    f = open(filename, 'w')
    headlines = ",".join(features) + "\n"
    f.write(headlines)
    for value in values:
        line = ",".join(value) + "\n"
        f.write(line)

    f.close()

def to_npy(csv_file):
    df = pd.read_csv(csv_file)
    X = np.array([])
    for i in tqdm(range(df.shape[0])):
        line = df.iloc[i]
        pid = int(line['PassengerId'])
        pclass = int(line['Pclass'])
        sex = 1 if line['Sex'] == 'male' else 0
        name = line['Name']
        if sex:
            if 'Mr.' in name:
                name = 1
            else:
                name = 0
        elif sex == 0:
            if 'Mrs.' in name:
                name = 1
            else:
                name = 0
        age = -1 if np.isnan(line['Age']) else int(line['Age'])
        sibsp = int(line['SibSp'])
        if pclass == 1:
            p = 80.
        elif pclass == 2:
            p = 25.
        elif pclass == 3:
            p = 10.
        parch =int(line['Parch'])
        fare =  p if np.isnan(line['Fare']) else int(line['Fare'])
        cabin = 1  if isinstance(line['Cabin'], str) else 0
        if line['Embarked'] == 'S':
            embarked = 0
        elif line['Embarked'] == 'C':
            embarked = 1
        elif line['Embarked'] == 'Q':
            embarked = 2

        X = np.append(X, [pid, pclass, name, sex, age, sibsp, parch, fare, cabin, embarked])
    return X.reshape(df.shape[0], 10)

def predict(X, male_model, female_model):
    features = [u'PassengerId',
                u'Survived']
    values = []
    for i in range(X.shape[0]):
        survived = male_model.predict(X[i, [1, 2, 6, 7, 8, 9]].reshape(1, -1)) if X[i, 3] == 1 else female_model.predict(X[i, [1, 2, 6, 7, 8, 9]].reshape(1, -1))
        values.append([str(int(X[i, 0])), str(int(survived))])
    create_csv_file('submission.csv', features, values)


if __name__ == '__main__':
    X = to_npy('data/CSV_FILES/test.csv')
    predict(X, pickle.load(open('data/Models/lgr_male.pkl', 'rb')), pickle.load(open('data/Models/xgb_female.pkl', 'rb')))
