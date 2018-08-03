import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm

def age_normalizer(X, col):
    X1 = normalize(X[:, col].reshape(-1, 1), axis=0)
    return np.concatenate((X[:, :col], X1, X[:, col+1:]), axis=1)

def to_npy(csv_file):
    df = pd.read_csv(csv_file)
    X = np.array([])
    y =  np.array([])
    for i in tqdm(range(df.shape[0])):
        line = df.iloc[i]
        survived = int(line['Survived'])
        pclass = int(line['Pclass'])
        sex = 1 if line['Sex'] == 'male' else 0
        age = -1 if np.isnan(line['Age']) else int(line['Age'])
        sibsp = int(line['SibSp'])
        parch = int(line['Parch'])
        fare = int(line['Fare'])
        cabin = 1  if isinstance(line['Cabin'], basestring) else 0

        if line['Embarked'] == 'S':
            embarked = 0
        elif line['Embarked'] == 'C':
            embarked = 1
        elif line['Embarked'] == 'Q':
            embarked = 2

        X = np.append(X, [pclass, sex, age, sibsp, parch, fare, cabin, embarked])
        y = np.append(y, survived) 
    X = X.reshape(df.shape[0], 8)
    X = normalize(X, axis=0)
    assert(X.shape[0] == y.shape[0])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42) 
    np.save('data/NPY_FILES/X_train', X_train)
    np.save('data/NPY_FILES/y_train', y_train)
    np.save('data/NPY_FILES/X_val', X_val) 
    np.save('data/NPY_FILES/y_val', y_val)
    

def to_npy_male_female(csv_file):
    df = pd.read_csv(csv_file)
    X_male = np.array([])
    X_female = np.array([])
    y_male = np.array([])
    y_female = np.array([])
    nb_male = 0
    nb_female = 0
    for i in tqdm(range(df.shape[0])):
        line = df.iloc[i]
        survived = int(line['Survived'])
        pclass = int(line['Pclass'])
        sex = 1 if line['Sex'] == 'male' else 0
        age = -1 if np.isnan(line['Age']) else int(line['Age'])
        sibsp = int(line['SibSp'])
        parch = int(line['Parch'])
        fare = int(line['Fare'])
        cabin = 1  if isinstance(line['Cabin'], basestring) else 0

        if line['Embarked'] == 'S':
            embarked = 0
        elif line['Embarked'] == 'C':
            embarked = 1
        elif line['Embarked'] == 'Q':
            embarked = 2
        
        if sex == 1:
            nb_male += 1
            X_male = np.append(X_male, [pclass, age, sibsp, parch, fare, cabin, embarked])
            y_male = np.append(y_male, survived)
        else:
            nb_female += 1
            X_female = np.append(X_female, [pclass, age, sibsp, parch, fare, cabin, embarked])
            y_female = np.append(y_female, survived)

    X_male = X_male.reshape(nb_male, 7)
#    X_male = normalize(X_male, axis=0)  Normalizing data decreases the accuracy on validation data 
    X_female = X_female.reshape(nb_female, 7)
#    X_female = normalize(X_female, axis=0)
    Xm_train, Xm_val, ym_train, ym_val = train_test_split(X_male, y_male, test_size=0.4, random_state=42) 
    Xf_train, Xf_val, yf_train, yf_val = train_test_split(X_female, y_female, test_size=0.4, random_state=42) 
    np.save('data/NPY_FILES/Xm_train', Xm_train)
    np.save('data/NPY_FILES/ym_train', ym_train)
    np.save('data/NPY_FILES/Xm_val', Xm_val) 
    np.save('data/NPY_FILES/ym_val', ym_val)
    np.save('data/NPY_FILES/Xf_train', Xf_train)
    np.save('data/NPY_FILES/yf_train', yf_train)
    np.save('data/NPY_FILES/Xf_val', Xf_val) 
    np.save('data/NPY_FILES/yf_val', yf_val)

if __name__ == '__main__':
    to_npy('data/CSV_FILES/train.csv')
    to_npy_male_female('data/CSV_FILES/train.csv')
