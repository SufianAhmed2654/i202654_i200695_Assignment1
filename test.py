import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Assuming the provided code is in a file named iris_model.py
from i202654_i200695 import iris_df, X_train, X_test, y_train, y_test, model

class TestIrisModel(unittest.TestCase):

    def test_data_loading(self):
        # Test if iris dataset is loaded correctly
        self.assertIsInstance(iris_df, pd.DataFrame)
        self.assertEqual(iris_df.shape[0], 150) # Iris dataset has 150 samples

    def test_data_preprocessing(self):
        # Test if 'species' column is correctly converted to categorical
        self.assertEqual(iris_df['species'].dtype, 'object')
        self.assertEqual(iris_df['species'].nunique(), 3)

    def test_train_test_split(self):
        # Test if train and test sets are correctly split
        self.assertEqual(len(X_train), 120)
        self.assertEqual(len(X_test), 30)

    def test_model_training(self):
        # Test if model is trained without errors
        self.assertIsInstance(model, LogisticRegression)
        self.assertGreater(model.score(X_test, y_test), 0.8) # Assuming a decent score

    def test_model_evaluation(self):
        # Test if model evaluation metrics are as expected
        score = model.score(X_test, y_test)
        self.assertGreater(score, 0.8) # Assuming a decent score

    def test_model_saving(self):
        # Test if model is saved without errors
        try:
            pickle.dump(model, open('model.pkl', 'wb'))
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Model saving failed: {e}")

if __name__ == '__main__':
    unittest.main()
