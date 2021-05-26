import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


def wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def get_all_k_metrics(total_pred):
    model_names = total_pred.columns[1:]
    index = ['MAE@5', 'MAE@10', 'MAE@20', 'wMAPE@5', 'wMAPE@10', 'wMAPE@20']
    final_res = pd.DataFrame(columns=model_names, index=index)
    for model in model_names:
        for k in [5, 10, 20]:
            pred_at_k = total_pred.iloc[total_pred['Real Quantity'].abs().argsort()][-k:]
            mae_k = mean_absolute_error(pred_at_k['Real Quantity'], pred_at_k[model])
            final_res.loc["MAE@" + str(k), model] = int(round(mae_k))
            wmape_k = wmape(pred_at_k['Real Quantity'], pred_at_k[model])
            final_res.loc["wMAPE@" + str(k), model] = (round(wmape_k, 3))
    return final_res
