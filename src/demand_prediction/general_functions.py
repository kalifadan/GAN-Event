import pandas as pd
import os
import os.path as path
import src.config as proj_config

cache_path = proj_config.CACHE_DIR
data_path = proj_config.DATA_DIR
events_data_path = proj_config.EVENTS_DATASET_DIR
categories_path = cache_path + '/categories_events/'


def get_file_path(name):
    return data_path + '/datasets/' + name + '.pkl'


def get_df_table(name):
    path_str = get_file_path(name)
    df = None
    if path.isfile(path_str):
        df = pd.read_pickle(path_str)
        print("Total data size: ", len(df))
    return df


def load_table_cache(name):
    path_str = cache_path + '/' + name
    df = None
    if path.isfile(path_str):
        df = pd.read_pickle(path_str)
    return df


def save_table_cache(df, name):
    path_str = cache_path + '/' + name
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    df.to_pickle(path_str)


def get_pred_dates(pred_start_date, pred_end_date):
    return [str(d).split()[0] for d in pd.date_range(start=pred_start_date, end=pred_end_date, freq='30D').tolist()]



