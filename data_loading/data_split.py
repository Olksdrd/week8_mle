import numpy as np
import os
import logging

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename='history.log',
    level=logging.INFO,
    format='%(asctime)s:%(module)s:%(levelname)s:%(message)s'
)


def load_dataset():
    logging.info('Loading dataset...')
    df = load_iris()
    X = df['data']
    y = df['target']

    return X, y


def split_traintest(X, y):
    logging.info('Splitting train and inference sets...')
    X_train, X_inference, y_train, y_inference = train_test_split(
        X, y,
        random_state=42,
        stratify=y,
        train_size=0.8
    )
    train_data = np.hstack([X_train, y_train.reshape(-1, 1)])
    inference_data = np.hstack([X_inference, y_inference.reshape(-1, 1)])

    return train_data, inference_data


def save_datasets(train_data, inference_data):
    CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../data'))
    
    logging.info('Creating "data" directory...')
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

    logging.info('Saving datasets...')
    np.savetxt('data/train_data.csv', train_data, delimiter=',', fmt='%10.4f')
    np.savetxt('data/inference_data.csv', inference_data, delimiter=',', fmt='%10.4f')


def main():
    logging.info(f'Starting {os.path.basename(__file__)} script...')
    X, y = load_dataset()
    train_data, inference_data = split_traintest(X, y)
    save_datasets(train_data, inference_data)
    logging.info(f'{os.path.basename(__file__)} execution finished.\n' + '-'*40)


if __name__ == '__main__':
    main()