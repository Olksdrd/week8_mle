import numpy as np
import os
import pickle

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_validate

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
MODEL_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../model'))
print(MODEL_DIRECTORY)
if not os.path.exists(MODEL_DIRECTORY):
    os.makedirs(MODEL_DIRECTORY)

df = np.loadtxt('data/train_data.csv',
                delimiter=",", dtype=float)
X = df[:, :-1]
y = df[:, -1]

logistic = LogisticRegressionCV(max_iter=500)

cv_results = cross_validate(logistic, X, y, cv=5)
print(cv_results['test_score'])

logistic.fit(X, y)
path = os.path.join(MODEL_DIRECTORY, 'model1.pickle')
with open(path, 'wb') as f:
            pickle.dump(logistic, f)