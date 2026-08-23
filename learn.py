import numpy as np
import logging
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor

# This is the main method to learn the probability distributions
# Takes P(m) and will then predict P(n)_pre and P(n)_post
def learn_probabilities(lists_of_pm, lists_of_n, learning_mode, pdml_mlp_options, pdml_ridge_options):
    logging.info(f"ML Learn Probabilities called with {learning_mode}")
    p_m = np.array(lists_of_pm)
    p_n_list = np.array(lists_of_n)
    x,y = p_m, p_n_list
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)
    # Assess using Multi Layer Perceptron
    if learning_mode == "mlp":
        clf = MLPRegressor(**pdml_mlp_options)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
    # Assess using Linear Regression
    elif learning_mode == "linear":
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict (x_test)
    # Assess using Ridge Regression
    elif learning_mode == "ridge":
        model = Ridge(**pdml_ridge_options)
        model.fit(x_train, y_train)
        y_pred = model.predict (x_test)
    else:
        raise ValueError(f"learn_probabilities must be called with mlp, linear or ridge actually got: {learning_mode}")

    mse = metrics.mean_squared_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    logging.info(f"ML Regression: {learning_mode}, MSE: {mse}, R2: {r2}")
    return y_test, y_pred, mse, r2

# This is the main method to learn the values of detuning from calculated (m)
# Given poor results this method needs to be augmented and updated.
def learn_detuning(lists_of_all_m, lists_of_detuning, learning_mode, cml_mlp_options):
    x,y = lists_of_all_m, lists_of_detuning
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)
    # Assess using Multi Layer Perceptron
    if learning_mode == "mlp":
        clf = MLPRegressor(**cml_mlp_options)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
    else:
        raise ValueError(f"learn_probabilities must be called with mlp actually got: {learning_mode}")
    mse = metrics.mean_squared_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    logging.info(f"ML Regression: {learning_mode}, MSE: {mse}, R2: {r2}")
    return y_test, y_pred, mse, r2
