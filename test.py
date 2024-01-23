import unittest
import logging

import numpy as np
import torch

import inference.prediction as prediction


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(module)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('history.log'),
        logging.StreamHandler()
    ]
)


class TestPrediction(unittest.TestCase):

    def test_load_inference_data(self):
        X, y = prediction.load_inference_data()
        self.assertEqual(np.isnan(np.sum(X)), 0)
        self.assertEqual(np.isnan(np.sum(y)), 0)
        self.assertTrue(X.shape[0] > 0)
        self.assertTrue(X.shape[1] == 4)
        self.assertTrue(y.ndim == 1)
        self.assertTrue(len(np.unique(y)) <= 3)
        logging.info('"load_inference_data" tested.')

    def test_data_to_tensor(self):
        X = np.array([[1, 2, 3, 4], [6, 7, 8, 9]])
        y = np.array([5, 10])
        X_test_tensor, y_test_tensor = prediction.data_to_tensor(X, y)
        self.assertTrue(torch.is_tensor(X_test_tensor))
        self.assertTrue(torch.is_tensor(y_test_tensor))
        self.assertEqual(X_test_tensor.dtype, torch.float32)
        self.assertEqual(y_test_tensor.dtype, torch.int64)
        logging.info('"data_to_tensor" tested.\n' + '-'*40)


if __name__ == '__main__':
    unittest.main()