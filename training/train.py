import numpy as np
import os
import pickle
import logging

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(
    filename='history.log',
    level=logging.INFO,
    format='%(asctime)s:%(module)s:%(levelname)s:%(message)s'
)

def load_training_data():
    logging.info('Loading training data...')
    df = np.loadtxt('data/train_data.csv',
                    delimiter=',', dtype=float)
    X = df[:, :-1]
    y = df[:, -1]
    return X, y


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, X):
        preds = self.linear(X)
        return preds


def train_model(X, y):
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y), dtype=torch.int32).reshape(-1, 1)
    y_train_tensor = y_train_tensor.type(torch.LongTensor)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor) # create your datset
    train_loader = DataLoader(train_dataset, batch_size = 5)

    model = Softmax()
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
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
    CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../model'))
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