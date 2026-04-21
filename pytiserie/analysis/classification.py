from sklearn.preprocessing import StandardScaler, pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import altair as alt


CONFIGURATION = {
    "regressor": MLPRegressor,
    "regressor_params": {
        "n_layers": 100,
        "n_outputs": 1,
        "n_feature_in": 10,
    },
    "features": [
        "crisp", "crunch", "color"
    ]
}


def load_data():
    df = pd.read_csv("/data/croissant_data.csv")
    df = df[CONFIGURATION['features']]
    return df


def fit_regressor(data):
    y = data['enjoyment']
    x = data.drop(columns=['enjoyment']).values

    model = CONFIGURATION['regressor'](**CONFIGURATION['regressor_params'])
    x_train, x_test, y_train, y_test = train_test_split(
                                                        x, y
    )
    model.fit(x, y)
    score = model.score(x_test, y_test)
    return score, model