import os
import sys
import logging
from time import time

import numpy as np
import torch

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
    """Loads data for inference."""
    try:
        logging.info('Loading data for inference...')
        df = np.loadtxt('data/inference_data.csv', delimiter=',', dtype=float)
        X = df[:, :-1]
        y = df[:, -1]
        logging.info(f'Inference set size is {X.shape[0]}')
        return X, y
    except Exception:
        logging.exception('Failed to load inference data.')


def data_to_tensor(X, y):
    """Converts inference data to tensors."""
    logging.info('Converting data to tensors...')
    X_test_tensor = torch.tensor(X, dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)
    y_test_tensor = y_test_tensor.type(torch.LongTensor)

    return X_test_tensor, y_test_tensor


def load_model():
    """Loads pre-trained model."""
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
    """Gets predicted classes for the evaluation data."""
    logging.info('Getting predictions...')
    pred_model = model(X)
    _, y_preds = pred_model.max(axis=1)
    return y_preds


def evaluate_predictions(y_true, y_pred):
    """Canculates accuracy score."""
    correct = (y_true.squeeze() == y_pred).sum().item()
    acc = correct / len(y_true)
    logging.info(f'Model accuracy = {acc:.3f}')
    return acc


def save_predictions(preds):
    """Saves predicted classes in results directory in csv format."""
    logging.info('Creating "results" directory...')
    if not os.path.exists(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)

    logging.info('Saving predictions...')
    np.savetxt('results/res.csv', preds, delimiter=',', fmt='%i')


def main():
    start_time = time()
    logging.info(f'Starting {os.path.basename(__file__)} script...')

    X, y = load_inference_data()
    X_test_tensor, y_test_tensor = data_to_tensor(X, y)
    ann = load_model()
    predictions = get_predictions(ann, X_test_tensor)
    evaluate_predictions(y_test_tensor, predictions)
    save_predictions(predictions)
    
    finish_time = time()
    time_delta = finish_time - start_time
    logging.info(f'Script execution took {time_delta:.2f} seconds.')
    logging.info(f'{os.path.basename(__file__)} execution finished.\n' + '-'*40)


if __name__ == '__main__':
    main()