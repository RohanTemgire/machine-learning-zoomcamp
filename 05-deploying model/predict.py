
# ### Loading the model

import pickle


input_file = 'model_C=1.0.bin'
# readbinary
with open(input_file, 'rb') as f_in:
    dv,model = pickle.load(f_in)


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
    'totalcharges': 29.85
}


def predict(cust):
    X = dv.transform([cust])
    y_pred =  model.predict_proba(X)
    return y_pred[0,1]

print(predict(customer), customer)

