import requests

url = 'http://127.0.0.1:9696/predict'


customer = {
    'gender' : 'female',
    'seniorcitize':0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines':
    'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': (1 * 29.85)
}

# print(json.dumps(customer))
response = requests.post(url, json=customer) ## post the customer information in json format
result = response.json() ## get the server response
print(result)