import pandas as pd
import src.config as proj_config
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from src.demand_prediction.events_models import calc_events_ts
from src.demand_prediction.general_functions import load_table_cache, save_table_cache

cache_path = proj_config.CACHE_DIR
data_path = proj_config.DATA_DIR
events_data_path = proj_config.EVENTS_DATASET_DIR
categories_path = cache_path + '/categories_events/'


def add_event_to_df(df, events, start_pred_time, leaf_name, window_size=3, use_cache=False, set_name='train', w=False):
    path_res = '/saved_df/' + leaf_name + '_' + set_name + '_' + start_pred_time + '.pkl'
    res = load_table_cache(path_res)
    if res is None or not use_cache:
        res = calc_events_ts(df, events, n=window_size, w=w)
        save_table_cache(res, path_res)
    return res


def split_data(data, start_pred_time, verbose=True):
    X_train = data[data['date'] < start_pred_time]
    X_test = data[data['date'] >= start_pred_time]
    if verbose:
        print("Train size:", len(X_train), " - Test size:", len(X_test))
    return X_train, X_test


def create_events_df(data, events, emb_only=False, emb_size=100):
    data.index = data.index.astype(str)
    data = data.reset_index()
    events['date'] = events['date'].astype(str)
    data = data.merge(events, on='date', how='left').dropna()
    data = data[['date', 'Category', 'embedding', 'wiki_name',
                 'High-Category', 'country', 'ref_num']]
    for ii in range(emb_size):
        col = 'emb_' + str(ii)
        data[col] = data['embedding'].apply(lambda x: x[ii])
    if not emb_only:
        one_hot = pd.get_dummies(data['Category'])
        data = data.join(one_hot)
        one_hot = pd.get_dummies(data['High-Category'])
        data = data.join(one_hot)
        one_hot = pd.get_dummies(data['country'])
        data = data.join(one_hot)
    return data.sort_values(by=['date']).reset_index(drop=True)


