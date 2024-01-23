import os
import sys
import logging
from time import time

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
    """Loads training dataset."""
    try:
        logging.info('Loading training data...')
        df = np.loadtxt('data/train_data.csv',
                        delimiter=',', dtype=float)
        X = df[:, :-1]
        y = df[:, -1]
        logging.info(f'Training set size is {X.shape[0]}')
        return X, y
    except Exception:
        logging.exception('Failed to load training data.')


def determine_batch_size(X):
    """Determines batch size based on training set size."""
    batch_vals = list(range(10, 1, -1))
    res = [X.shape[0] % i for i in range(10, 1, -1)]
    return batch_vals[res.index(0)]


def data_to_tensor(X, y):
    """Loads training data into dataloader."""
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
    """Returns trained model."""
    train_loader = data_to_tensor(X, y)

    model = Softmax()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    loss_vals = []
    epochs = 100
    start_time = time()
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.squeeze())
            loss_vals.append(loss)
            loss.backward()
            optimizer.step()
    finish_time = time()
    time_delta = finish_time - start_time
    logging.info(f'Model training took {time_delta:.2f} seconds.')
    return model


def save_model(model):
    """Saves the model in .pth format in model folder."""
    logging.info('Creating "model" directory...')
    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    logging.info('Saving the model...')
    path = os.path.join(MODEL_DIRECTORY, 'logistic.pth')
    torch.save(model.state_dict(), path)


def main():
    start_time = time()
    logging.info(f'Starting {os.path.basename(__file__)} script...')

    X, y = load_training_data()
    ann = train_model(X, y)
    save_model(ann)
    finish_time = time()
    
    time_delta = finish_time - start_time
    logging.info(f'Script execution took {time_delta:.2f} seconds.')
    logging.info(f'{os.path.basename(__file__)} execution finished.\n' + '-'*40)


if __name__ == '__main__':
    main()