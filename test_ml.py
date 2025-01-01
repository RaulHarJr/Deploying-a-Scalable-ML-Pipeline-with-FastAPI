import pandas as pd
import numpy as np
import pytest
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml.model import train_model, inference, compute_model_metrics

# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_is_algorithm():
    """
    Test if the algorithm used by train_model is RandomForestClassifier
    """
    X, y = make_classification(n_samples=200, n_features=5, n_informative=2,
                               n_redundant=0, n_classes=2, random_state=42)
    
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_in_shape():
    """
    Test if the inference is producing the correct shape 
    """
    X, y = make_classification(n_samples=200, n_features=5, n_informative=2,
                               n_redundant=0, n_classes=2, random_state=42)
    
    
    model = train_model(X, y)
    predictions = inference(model, X)

    assert predictions.shape[0] == X.shape[0]


# TODO: implement the third test. Change the function name and input as needed
def test_metrics():
    """
    Test to see if compute metrics is giving correct Precision, Recall and F1beta
    """
    # We'll use 2 lists of 20 values, 18 match. 9 out of 10 1's match
    A_vals = [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    B_vals = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    precision, recall, f1beta = compute_model_metrics(A_vals, B_vals)

    assert precision == pytest.approx(0.9, rel=1e-10)
    assert recall  == pytest.approx(0.9, rel=1e-10)
    assert f1beta == pytest.approx(0.9, rel=1e-10)

