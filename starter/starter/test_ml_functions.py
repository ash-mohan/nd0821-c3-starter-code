from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

import pytest
import numpy as np
import pandas as pd
import sys

sys.path.insert(1, './starter/ml')
sys.path.append('./starter/starter/ml')

try:
    from data import process_data
    from model import train_model, compute_model_metrics, inference
except Exception as e:
    raise e


@pytest.fixture(scope="session")
def data():

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    data = pd.read_csv("starter/data/census_data_clean.csv")
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    return X_train, X_test, y_train, y_test, encoder, lb


def test_process_data(data):
    """ Test to ensure that the process data function works properly"""

    X_train, X_test, y_train, y_test, encoder, lb = data

    # make sure return types are valid
    assert type(X_train) == np.ndarray
    assert type(X_test) == np.ndarray
    assert type(y_train) == np.ndarray
    assert type(y_test) == np.ndarray
    assert type(encoder) == OneHotEncoder
    assert type(lb) == LabelBinarizer

    # train and test data should have same features
    assert X_train.shape[1] == X_test.shape[1]

    # there should be two classes
    assert len(lb.classes_) == 2


def test_train_model(data):
    """ Test to ensure that train model function returns correct model """

    X_train = data[0]
    y_train = data[2]

    model = train_model(X_train, y_train)

    assert type(model) == GradientBoostingClassifier


def test_model_metrics(data):
    """ Tests to ensure that model metrics are valid"""

    # we will check that the range is valid
    X_train, _, y_train, _, _, _ = data
    model = train_model(X_train, y_train)
    y_pred = inference(model, X_train)
    precision, recall, fbeta, accuracy = compute_model_metrics(y_train, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
    assert 0 <= accuracy <= 1
