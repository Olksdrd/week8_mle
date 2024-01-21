import numpy as np
import os
import pickle
import logging

logging.basicConfig(
    filename='history.log',
    level=logging.INFO,
    format='%(asctime)s:%(module)s:%(levelname)s:%(message)s'
)

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../model'))
RESULTS_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../results'))


def load_inference_data():
    logging.info('Loading data for inference...')
    df = np.loadtxt('data/inference_data.csv', delimiter=',', dtype=float)
    X = df[:, :-1]
    y = df[:, -1]

    return X, y

def load_model():
    logging.info('Loading the model...')
    path = os.path.join(MODEL_DIRECTORY, 'model1.pickle')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def get_predictions(model, X):
    logging.info('Getting predictions...')
    preds = model.predict(X)
    return preds


def save_predictions(preds):
    logging.info('Creating "results" directory...')
    if not os.path.exists(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)

    logging.info('Saving predictions...')
    np.savetxt('results/res.csv', preds, delimiter=',', fmt='%i')


def main():
    logging.info(f'Starting {os.path.basename(__file__)} script...')
    X, _ = load_inference_data()
    model = load_model()
    res = get_predictions(model, X)
    save_predictions(res)
    logging.info(f'{os.path.basename(__file__)} execution finished.\n' + '-'*40)


if __name__ == '__main__':
    main()