import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction import DictVectorizer 
from sklearn.metrics import roc_auc_score
import pickle
import xgboost as xgb

# parameters
# 0.2    9  100  200 -> 0.921
e = 0.2
d = 9
s = 100
nbr = 200
output_file = f"xgb_model.bin"


# data prepartion
print("-------------- preparing the data --------------")
df = pd.read_csv('train.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

df['loan_amount'] = np.log1p(df.loan_amount)
df['annual_income'] = np.log1p(df.annual_income)


# splitting the datasets

print("-------------- splitting the datasets --------------")
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=15)

y_train = df_full_train['loan_paid_back']
y_test = df_test['loan_paid_back']

del df_full_train['loan_paid_back']
del df_test['loan_paid_back']

del df_full_train['id']
del df_test['id']



# training the model
print("-------------- training the model --------------")



def train_xgb_model(df, y, e=0.3, d=10, s=100, nbr=100):
    train_dicts = df.to_dict(orient = 'records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label = y, feature_names = features)

    xgb_params = {
        'eta': e, 
        'max_depth': d,
        'min_child_weight': s,
        
        'objective': 'binary:logistic',
        'nthread': 8,
        
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=nbr)
    return model, dv


def test_xgb_model(model, dv, x, y):
    features = list(dv.get_feature_names_out())

    X_test = dv.transform(x.to_dict(orient='records'))

    dtest = xgb.DMatrix(X_test, label = y, feature_names = features)

    y_pred = model.predict(dtest)
    return y_pred

# training the final model

print("-------------- training the final model --------------")

model, dv = train_xgb_model(df_full_train, y_train, e=e, d=d, s=s, nbr=nbr)
y_pred = test_xgb_model(model, dv, df_test, y_test)


auc = roc_auc_score(y_test, y_pred)
print(f'------------- the score is {auc} ----------')

# ### saving the model

# 'wb' -> writebinary

print("-------------- saving the model --------------")
with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)

print(f"-------------- model saved to {output_file} --------------")
