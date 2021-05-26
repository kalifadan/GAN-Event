import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models.rnn_model import RNNModel
from darts.dataprocessing.transformers import Scaler
from src.demand_prediction.dataset_functions import add_event_to_df
from src.demand_prediction.events_models import save_events_model, load_events_model
from src.demand_prediction.gan_events import get_gan_embeddings
from src.demand_prediction.event_lstm import EventRNNModel
from src.config import SEED


def lstm_model_results(train, test, train_df, test_df, events_all, start_pred_time, leaf_name, n_in,
                       model='LSTM', window_size=2, categ_data=None, device='cuda:2', lstm_df_cache=False, use_cache=False):
    train_set, test_set, test_ts = None, None, None

    if model == 'Event LSTM':
        train_set = add_event_to_df(train_df, events_all, start_pred_time, leaf_name, window_size=window_size,
                                    use_cache=lstm_df_cache, set_name='train')
        train_set = train_set[['emb_' + str(ii) for ii in range(100)]]
        test_set = add_event_to_df(test_df, events_all, start_pred_time, leaf_name, window_size=window_size,
                                   use_cache=lstm_df_cache, set_name='test')

    if model == 'Weighted Event LSTM':
        train_set = add_event_to_df(train_df, events_all, start_pred_time, leaf_name + " - wLSTM", window_size=window_size,
                                    use_cache=False, set_name='train', w=lstm_df_cache)
        train_set = train_set[['emb_' + str(ii) for ii in range(100)]]
        test_set = add_event_to_df(test_df, events_all, start_pred_time, leaf_name + " - wLSTM", window_size=window_size,
                                   use_cache=False, set_name='test', w=lstm_df_cache)

    if model == 'GAN - Event LSTM':
        train_set, test_set = get_gan_embeddings(train_df, test_df, categ_data, window_size=window_size)
        train_set = train_set[['emb_' + str(ii) for ii in range(100)]]

    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(test)

    if not model == 'LSTM':
        train_transformed = TimeSeries(train_transformed.pd_dataframe().merge(train_set, on='date'))
        test_set = test_set[['emb_' + str(ii) for ii in range(100)]]
        test_transformed = test_transformed.pd_dataframe().merge(test_set, on='date')
        test_ts = train_transformed[-n_in:]

    test_df = test.pd_dataframe()
    y_test = np.array(test_df.Quantity.values)
    name_path_model = leaf_name + "_" + model + "_" + start_pred_time
    lstm_model = load_events_model(name_path_model) if use_cache else None

    if lstm_model is None or not use_cache:
        if not model == 'LSTM':
            print("Event LSTM Model")
            lstm_model = EventRNNModel(
                model='LSTM',
                n_epochs=100,
                hidden_size=404,
                input_chunk_length=n_in,
                output_chunk_length=1,
                dropout=0.3,
                optimizer_kwargs={'lr': 0.001},
                log_tensorboard=True,
                torch_device_str=device,
                model_name=name_path_model,
                random_state=SEED)

        else:
            print("LSTM Model")
            lstm_model = RNNModel(
                model='LSTM',
                n_epochs=100,
                input_chunk_length=n_in,
                output_chunk_length=1,
                dropout=0.3,
                optimizer_kwargs={'lr': 0.001},
                log_tensorboard=True,
                torch_device_str=device,
                model_name=name_path_model,
                random_state=SEED)

        lstm_model.fit(train_transformed, verbose=True)
        save_events_model(lstm_model, name_path_model)

    if model == 'LSTM':
        forecast = lstm_model.predict(len(test))

    else:
        prediction_time, forecast = len(test), None
        for ii in range(prediction_time):
            cur_forecast = lstm_model.predict(1, test_ts)
            next_day = test_transformed.iloc[ii:ii+1]
            next_day['0'] = cur_forecast.values()[0][0]
            next_day = TimeSeries.from_dataframe(next_day, freq="D")
            test_ts = test_ts[1:].append(next_day)
            forecast = cur_forecast if forecast is None else forecast.append(cur_forecast)

    predictions = forecast.pd_dataframe().rename(columns={'0': model})
    predictions = transformer.inverse_transform(TimeSeries.from_dataframe(predictions, freq="D")).pd_dataframe()
    predictions = predictions.rename(columns={'0': model})

    y_test_df = pd.DataFrame(test_df, index=test_df.index, columns=['Quantity']).rename(columns={'Quantity': 'Real Quantity'})
    df_test_results = pd.concat([y_test_df, predictions], axis=1)
    df_test_results.iplot(title=leaf_name + " - " + model, xTitle='Date', yTitle='Sales', theme='white')

    return predictions


def get_lstm_results(train, test, train_df, test_df, events_all, start_pred_time, leaf_name, n_in,
                     window_size, categ_data, device='cuda:2', lstm_df_cache=False, use_cache=False):
    lstm_predictions = []
    for model in ['LSTM', 'Event LSTM', 'GAN - Event LSTM', 'Weighted Event LSTM']:
        predictions = lstm_model_results(train, test, train_df, test_df, events_all, start_pred_time, leaf_name,
                        n_in, model, window_size, categ_data, device=device, lstm_df_cache=lstm_df_cache, use_cache=use_cache)
        lstm_predictions.append(predictions)
    lstm_predictions = pd.concat(lstm_predictions, axis=1)
    return lstm_predictions
