from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICES'])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

property_stats = features.mean().values.reshape(1,11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_room,students_per_classroom, next_to_river=False, high_confidence=True):

    property_stats[0][RM_IDX] = nr_room
    property_stats[0][PTRATIO_IDX] = students_per_classroom

    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0

    log_estimate = regr.predict(property_stats)[0][0]

    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68

    return log_estimate, upper_bound, lower_bound, interval
