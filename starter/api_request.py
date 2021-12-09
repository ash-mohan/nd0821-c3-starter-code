import requests
import json

heroku_app = "https://devops-proj3.herokuapp.com/"
api_path = "inference/"

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

r = requests.post(heroku_app + api_path, data=json.dumps(request_body))

print(f"Request finished with {r.status_code} status")
print(r.content)
