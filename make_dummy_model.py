import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import io
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Pickle!
import joblib


def format_number(x):
    return '{:.2e}'.format(x)


def make_dummy_model():
    # test_data_lol = pd.read_csv("data/test.csv")

    # y_lol = pd.read_csv("data/sample_submission.csv")
    train_data_lol = pd.read_csv("data/train.csv")

    # Split the data into training/testing sets
    train, test = train_test_split(
        train_data_lol[['LotArea', 'BedroomAbvGr', 'YearBuilt', 'FullBath', 'HalfBath', 'SalePrice']], test_size=0.2,
        random_state=12123213)

    #
    # X_meh = train[['LotArea', 'BedroomAbvGr', 'YearBuilt', 'FullBath', 'HalfBath']]
    # y_meh = train_data_lol[['SalePrice']]

    reg = LinearRegression().fit(train.loc[:, train.columns != 'SalePrice'], train[['SalePrice']])
    reg.coef_

    rmse = mean_squared_error(test[['SalePrice']], reg.predict(test.loc[:, test.columns != 'SalePrice']), squared=False)
    print("MSE: ", format_number(rmse))

    with open('bad_model.pkl', 'wb') as fid:
        joblib.dump(reg, fid, compress=9)

    print("Done!")


make_dummy_model()
