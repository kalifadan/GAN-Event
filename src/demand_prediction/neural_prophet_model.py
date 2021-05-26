import pandas as pd
from neuralprophet import NeuralProphet, set_random_seed
from src.demand_prediction.events_models import save_events_model, load_events_model
from src.config import SEED


def NeuralProphetEvents(future_events, past_events, events_name, train, test, leaf_name, model_name,
                        start_pred_time, events_dates, use_cache=False):
    test_name = leaf_name
    test_df = test.pd_dataframe()
    train_df = train.pd_dataframe()
    train_df['ds'] = train_df.index
    train_df = train_df.rename(columns={'Quantity': 'y'})

    name_path_model = leaf_name + "_" + model_name + "_" + start_pred_time
    model = load_events_model(name_path_model)

    if model is None or not use_cache:
        print("Training Event Neural Prophet")
        set_random_seed(SEED)
        model = NeuralProphet()
        model = model.add_country_holidays("US", mode="additive", lower_window=-1, upper_window=1)
        model.add_events(events_name)
        history_df = model.create_df_with_events(train_df, past_events)
        print("Event Neural Prophet Fitting")
        metrics = model.fit(history_df, freq='D')
        save_events_model(model, name_path_model)
        save_events_model(history_df, name_path_model + "_history_df")
    else:
        print("Loaded Event Neural Prophet")
        history_df = load_events_model(name_path_model + "_history_df")
        if history_df is None:
            print("Creating History df Neural Prophet")
            history_df = model.create_df_with_events(train_df, past_events)
            save_events_model(history_df, name_path_model + "_history_df")

    print("Start Predicting:")
    future = model.make_future_dataframe(df=history_df, events_df=future_events, periods=len(test))
    forecast = model.predict(future)
    preds = forecast[['ds', 'yhat1']]
    predictions = pd.DataFrame(preds).rename(columns={'ds': "Date", 'yhat1': model_name})
    predictions.index = predictions.Date
    predictions = predictions.drop(columns=['Date'])
    y_test_df = pd.DataFrame(test_df, index=test_df.index, columns=['Quantity']).rename(
        columns={'Quantity': 'Real Quantity'})
    df_test_results = pd.concat([y_test_df, predictions], axis=1)
    df_test_results.iplot(title=test_name, xTitle='Date', yTitle='Sales', theme='white')
    return predictions


def reformat_events_name(events):
    return events[['date', 'wiki_name']].rename(columns={'date': 'ds', 'wiki_name': 'event'})


def get_events_for_neural_prophet(events_all, start_pred_time):
    train_events = events_all[events_all.date < start_pred_time]
    test_events = events_all[events_all.date >= start_pred_time]
    return reformat_events_name(train_events), reformat_events_name(test_events)


def get_neural_prophet_results(train, test, events_all, leaf_name, events_dates, start_pred_time,
                               use_cache=False):
    modes_name = {1: 'Event NeuralProphet'}
    cur_mode = 1
    past_opt_events, future_opt_events = get_events_for_neural_prophet(events_all, start_pred_time)
    print("Total future optimal events:", len(future_opt_events))
    print("Total past optimal events:", len(past_opt_events))
    events_name = list(set(list(future_opt_events.event.values) + list(past_opt_events.event.values)))
    predictions = NeuralProphetEvents(future_opt_events, past_opt_events, events_name, train, test,
                                              leaf_name, modes_name[cur_mode], start_pred_time, events_dates, use_cache)
    return predictions
