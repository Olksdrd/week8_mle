import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(module)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('history.log'),
        logging.StreamHandler()
    ]
)


CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../model'))
sys.path.append(os.path.dirname(CURRENT_DIRECTORY))


try:
    from utils import Softmax
except:
    logging.exception('Failed to load the Softmax class from utils.py.')


def load_training_data():
    try:
        logging.info('Loading training data...')
        df = np.loadtxt('data/train_data.csv',
                        delimiter=',', dtype=float)
        X = df[:, :-1]
        y = df[:, -1]
        return X, y
    except Exception:
        logging.exception('Failed to load training data.')


def determine_batch_size(X):
    batch_vals = list(range(10, 1, -1))
    res = [X.shape[0] % i for i in range(10, 1, -1)]
    return batch_vals[res.index(0)]


def data_to_tensor(X, y):
    batch_size = determine_batch_size(X)
    logging.info(f'Batch size set to {batch_size}.')

    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y), dtype=torch.int32).reshape(-1, 1)
    y_train_tensor = y_train_tensor.type(torch.LongTensor)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    logging.info('Data loaded into DataLoader.')

    return train_loader


def train_model(X, y):
    train_loader = data_to_tensor(X, y)

    model = Softmax()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    loss_vals = []
    epochs = 100
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.squeeze())
            loss_vals.append(loss)
            loss.backward()
            optimizer.step()
    
    return model


def save_model(model):
    logging.info('Creating "model" directory...')
    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    logging.info('Saving the model...')
    path = os.path.join(MODEL_DIRECTORY, 'logistic.pth')
    torch.save(model.state_dict(), path)


def main():
    logging.info(f'Starting {os.path.basename(__file__)} script...')
    X, y = load_training_data()
    ann = train_model(X, y)
    save_model(ann)
    logging.info(f'{os.path.basename(__file__)} execution finished.\n' + '-'*40)


if __name__ == '__main__':
    main()