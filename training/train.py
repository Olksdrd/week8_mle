import numpy as np
import os
import pickle
import logging

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_validate

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


def train_model(X, y):
    logging.info('Doing cross-validation...')
    model = LogisticRegressionCV(max_iter=500)
    cv_results = cross_validate(model, X, y, cv=5)
    logging.info(f'CV results: {np.mean(cv_results["test_score"]):.3f}')

    logging.info('Refitting the model...')
    model.fit(X, y)
    return model


def save_model(model):
    CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../model'))
    logging.info('Creating "model" directory...')
    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)

    logging.info('Saving the model...')
    path = os.path.join(MODEL_DIRECTORY, 'model1.pickle')
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def main():
    logging.info(f'Starting {os.path.basename(__file__)} script...')
    X, y = load_training_data()
    logistic = train_model(X, y)
    save_model(logistic)
    logging.info(f'{os.path.basename(__file__)} execution finished.\n' + '-'*40)



if __name__ == '__main__':
    main()