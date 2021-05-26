import os
import pickle
import pandas as pd
import src.config as proj_config
cache_path = proj_config.CACHE_DIR


def calc_events_ts(df, events, n=3, w=False):
    data = df.copy()
    data['date'] = data.index.astype('str')
    if not w:
        events = events.drop(columns=['Category', 'embedding', 'wiki_name', 'High-Category', 'country', 'ref_num'])
    else:
        events = events.drop(columns=['Category', 'embedding', 'wiki_name', 'High-Category', 'country'])
    res = df.copy()
    res_cols = []

    agg_dict = {}
    for col in events.columns:
        if col.startswith("emb_"):
            agg_dict[col] = 'mean'
            res[col] = 0
            res_cols.append(col)
        elif col == 'date':
            pass

    for row_idx, row in enumerate(data.values):
        date = row[data.columns.get_loc('date')]
        dates_range = [str(d).split()[0] for d in pd.date_range(end=date, periods=n, closed='right').tolist()]
        selected_events = events[events.date >= dates_range[0]]
        selected_events = selected_events[selected_events.date <= dates_range[-1]]

        if not selected_events.empty:
            if not w:
                selected_events = selected_events.drop(columns=['date']).agg(agg_dict)
            if w:
                selected_events = selected_events.drop(columns=['date'])
                selected_events['ref_num'] += 1  # for handle div by zero
                total_ref = sum(selected_events['ref_num'].values)
                ref_values = selected_events['ref_num'].values
                selected_events = selected_events.drop(columns=['ref_num']).mul(ref_values, axis=0)
                selected_events = selected_events.sum() / total_ref

            for ii, col in enumerate(res_cols):
                res.loc[date, col] = selected_events[ii]

    return res


def load_events_model(name):
    filename = cache_path + '/events_models/' + name + '.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        return None


def save_events_model(model, name):
    filename = cache_path + '/events_models/' + name + '.pkl'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump(model, open(filename, 'wb'))