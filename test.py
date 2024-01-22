import unittest
import logging

import numpy as np

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
        logging.info('"load_inference_data" tested.\n' + '-'*40)


if __name__ == '__main__':
    unittest.main()