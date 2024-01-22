import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn

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
RESULTS_DIRECTORY = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '../results'))
sys.path.append(os.path.dirname(CURRENT_DIRECTORY))


try:
    from utils import Softmax
except:
    logging.exception('Failed to load the Softmax class from utils.py.')


def load_inference_data():
    try:
        logging.info('Loading data for inference...')
        df = np.loadtxt('data/inference_data.csv', delimiter=',', dtype=float)
        X = df[:, :-1]
        y = df[:, -1]

        X_test_tensor = torch.tensor(X, dtype=torch.float32)
        y_test_tensor = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
        y_test_tensor = y_test_tensor.type(torch.LongTensor)

        return X_test_tensor, y_test_tensor
    except Exception:
        logging.exception('Failed to load inference data.')


def load_model():
    try:
        logging.info('Loading the model...')
        path = os.path.join(MODEL_DIRECTORY, 'logistic.pth')
        model = Softmax()
        model.load_state_dict(torch.load(path))
        model.eval()

        return model
    except Exception:
        logging.exception('Failed to load the model.')


def get_predictions(model, X):
    logging.info('Getting predictions...')
    pred_model = model(X)
    _, y_preds = pred_model.max(axis=1)
    return y_preds


def evaluate_predictions(y_true, y_pred):
    correct = (y_true.squeeze() == y_pred).sum().item()
    acc = correct / len(y_true)
    logging.info(f'Model accuracy = {acc:.3f}')
    return acc


def save_predictions(preds):
    logging.info('Creating "results" directory...')
    if not os.path.exists(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)

    logging.info('Saving predictions...')
    np.savetxt('results/res.csv', preds, delimiter=',', fmt='%i')


def main():
    logging.info(f'Starting {os.path.basename(__file__)} script...')
    X_test_tensor, y_test_tensor = load_inference_data()
    ann = load_model()
    predictions = get_predictions(ann, X_test_tensor)
    evaluate_predictions(y_test_tensor, predictions)
    save_predictions(predictions)
    logging.info(f'{os.path.basename(__file__)} execution finished.\n' + '-'*40)


if __name__ == '__main__':
    main()