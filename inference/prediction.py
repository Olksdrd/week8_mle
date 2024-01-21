import numpy as np
import os
import pickle

from sklearn.linear_model import LogisticRegressionCV

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
RESULTS_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../results'))
MODEL_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../model'))
print(RESULTS_DIRECTORY)
if not os.path.exists(RESULTS_DIRECTORY):
    os.makedirs(RESULTS_DIRECTORY)

df = np.loadtxt('data/inference_data.csv',
                delimiter=",", dtype=float)
X = df[:, :-1]
y = df[:, -1]

path = os.path.join(MODEL_DIRECTORY, 'model1.pickle')
with open(path, 'rb') as f:
     model = pickle.load(f)

res = model.predict(X)
print(res)

np.savetxt('results/res.csv', res, delimiter=',', fmt='%10.4f')