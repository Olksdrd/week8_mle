import numpy as np
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

df = load_iris()

X = df['data']
y = df['target']

X_train, X_inference, y_train, y_inference = train_test_split(
    X, y,
    random_state=42,
    stratify=y,
    train_size=0.8
)

train_data = np.hstack([X_train, y_train.reshape(-1, 1)])
inference_data = np.hstack([X_inference, y_inference.reshape(-1, 1)])

print(train_data.shape)
print(inference_data.shape)

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
DATA_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../data'))
print(CURRENT_DIRECTORY)
print(ROOT_DIRECTORY)
print(DATA_DIRECTORY)

if not os.path.exists(DATA_DIRECTORY):
    os.makedirs(DATA_DIRECTORY)

np.savetxt('data/train_data.csv', train_data, delimiter=',', fmt='%10.4f')
np.savetxt('data/inference_data.csv', inference_data, delimiter=',', fmt='%10.4f')