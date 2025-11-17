import pickle
from fastapi import FastAPI, HTTPException
import xgboost as xgb
import numpy as np

from cust_model import Customer



input_file = 'xgb_model.bin'
# readbinary
with open(input_file, 'rb') as f_in:
    dv,model = pickle.load(f_in)



app = FastAPI()


async def predict_single(cust,dv,model):
    try:

        cust['loan_amount'] = np.log1p(cust['loan_amount'])
        cust['annual_income'] = np.log1p(cust['annual_income'])

        features = list(dv.get_feature_names_out())
        X = dv.transform([cust])

        dtest = xgb.DMatrix(X, feature_names = features)
        
        y_pred =  model.predict(dtest)

        if int(y_pred >= 0.5):
            return "the loan can be given out safely to the customer"
        else:
            return "Giving loan to this customer is risky"
    except Exception as e:
        return HTTPException(status_code = 404, detail=f"error occured: {e}")




@app.get('/')
def home():
    return "Hello to the bank finance world"


@app.post('/predict') 
async def predict(cust: Customer):

    try: 
        customer = cust.dict()
        prediction = await predict_single(customer, dv, model)

        return {
            'messgae': prediction
        }
    except Exception as e:
        return HTTPException(status_code = 404, detail=f"error occured: {e}")
    # return jsonify(result) 
