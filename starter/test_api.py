from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome from inference API"}


def test_post1():

    request_body = {
        'age': 36,
        'workclass': 'Private',
        'fnlgt': 219814,
        'education': '9th',
        'education-num': 5,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Craft-repair',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 35,
        'native-country': 'Guatemala'
    }

    r = client.post("/inference/", data=json.dumps(request_body))

    assert r.status_code == 200
    assert r.json()["results"] == {
        "binary_class": [0],
        "class": ['<=50K']
    }


def test_post_2():

    request_body = {
        'age': 49,
        'workclass': 'Private',
        'fnlgt': 122385,
        'education': 'Masters',
        'education-num': 14,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'}

    r = client.post("/inference/", data=json.dumps(request_body))

    assert r.status_code == 200
    assert r.json()["results"] == {
        "binary_class": [1],
        "class": ['>50K']
    }
