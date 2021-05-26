import os
import pickle
import pandas as pd
import src.config as proj_config
from darts.models import ExponentialSmoothing
from darts.models import AutoARIMA
from darts.models.prophet import Prophet
from neuralprophet import NeuralProphet, set_random_seed
from src.config import SEED
cache_path = proj_config.CACHE_DIR


def train_models(train, model_name, verbose=True):
    if model_name == 'ARIMA':
        arima_model = AutoARIMA()
        arima_model.fit(train)
        return arima_model

    if model_name == 'Exponential Smoothing':
        ema_model = ExponentialSmoothing()
        ema_model.fit(train)
        return ema_model

    if model_name == 'Prophet':
        prophet_model = Prophet(country_holidays='US')
        prophet_model.fit(train)
        return prophet_model

    if model_name == 'NeuralProphet':
        set_random_seed(SEED)
        train_df = train.pd_dataframe()
        train_df['ds'] = train_df.index
        train_df = train_df.rename(columns={'Quantity': 'y'})
        neural_model = NeuralProphet()
        neural_model = neural_model.add_country_holidays("US", mode="additive", lower_window=-1, upper_window=1)
        metrics = neural_model.fit(train_df, freq='D')
        return neural_model

    return None


def test_models(test, test_name, start_pred_time, train, use_cache=True):
    test_df = test.pd_dataframe()
    total_pred = pd.DataFrame(test_df, index=test_df.index, columns=['Quantity']).rename(columns={'Quantity': 'Real Quantity'})
    Models = ['ARIMA', 'Prophet', 'NeuralProphet']

    for model_name in Models:
        print("Model: ", model_name)
        model = load_model(model_name, train, test_name + "_" + start_pred_time, use_cache, verbose=False)

        if not model_name == 'NeuralProphet':
            predictions = model.predict(len(test)).pd_dataframe().rename(columns={'0': model_name})

        else:
            train_df = train.pd_dataframe()
            train_df['ds'] = train_df.index
            train_df = train_df.rename(columns={'Quantity': 'y'})
            future = model.make_future_dataframe(train_df, periods=len(test))
            forecast = model.predict(future)
            preds = forecast[['ds', 'yhat1']]
            predictions = pd.DataFrame(preds).rename(columns={'ds': "Date", 'yhat1': model_name})
            predictions.index = predictions.Date
            predictions = predictions.drop(columns=['Date'])

        y_test_df = pd.DataFrame(test_df, index=test_df.index, columns=['Quantity']).rename(
            columns={'Quantity': 'Real Quantity'})
        df_test_results = pd.concat([y_test_df, predictions], axis=1)
        total_pred = pd.concat([total_pred, predictions], axis=1)
        cur_name = test_name + " - " + model_name
        df_test_results.iplot(title=cur_name, xTitle='Date', yTitle='Sales', theme='white')

    return total_pred


def save_model(model_to_save, model_name, leaf_name):
    filename = cache_path + '/saved_models/' + model_name + '_' + leaf_name + '.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(model_to_save, file)


def load_model(model_name, train, leaf_name, use_cache=True, verbose=False):
    filename = cache_path + '/saved_models/' + model_name + '_' + leaf_name + '.pkl'
    if os.path.isfile(filename) and use_cache:
        print("Loaded model from cache.")
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        model = train_models(train, model_name, verbose=verbose)
        save_model(model, model_name, leaf_name)
        return model