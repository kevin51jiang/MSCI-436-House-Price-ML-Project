import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import io

# Pickle!
import joblib


def make_dummy_model():
    test_data_lol = pd.read_csv("data/test.csv")
    y_lol = pd.read_csv("data/sample_submission.csv")
    train_data_lol = pd.read_csv("data/train.csv")

    X_meh = train_data_lol[['LotArea', 'BedroomAbvGr', 'YearBuilt', 'FullBath', 'HalfBath']]
    y_meh = train_data_lol[['SalePrice']]

    reg = LinearRegression().fit(X_meh, y_meh)
    reg.coef_

    with open('bad_model.pkl', 'wb') as fid:
        joblib.dump(reg, fid, compress=9)

    print("Done!")

make_dummy_model()

