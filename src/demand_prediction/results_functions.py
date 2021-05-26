import pandas as pd
import numpy as np


def wmape(y_true, y_pred):
    # based on "A Deep Neural Framework for Sales Forecasting in E-Commerce" paper
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# def get_metrics_at_k(k, total_pred, peaks_data, final_list, leaf_name, start_pred_list, prediction_time, metric=None):
#     pred_at_k = total_pred.iloc[total_pred['Real Quantity'].abs().argsort()][-k:]
#     df_k = get_combine_results(pred_at_k, peaks_data, final_list, leaf_name, start_pred_list, prediction_time, verbose=False)[['MAE', 'MAPE']]
#     df_k['MAE'] = df_k['MAE'].round().astype(int)
#     df_k.columns = df_k.columns.droplevel(1)
#
#     dcg_models, wmape_models = [], []
#     for model_name in df_k.index.values:
#         dcg_models.append(round(ndcg_metric(pred_at_k, model_name, k), 3))
#         wmape_models.append(round(wmape(pred_at_k['Real Quantity'].values, pred_at_k[model_name].values), 3))
#
#     df_k['NDCG'] = dcg_models
#     df_k['wMAPE'] = wmape_models
#
#     return df_k
#
#
# def get_all_k_metrics(total_pred, peaks_data, final_list, leaf_name, start_pred_list, prediction_time):
#     all_df_k = []
#     for ii, k in enumerate([5, 10, 20, 25, len(total_pred)]):
#         k_name = ['@5', '@10', '@20', '@25', ''][ii]
#         all_df_k.append(get_metrics_at_k(k, total_pred, peaks_data, final_list, leaf_name, start_pred_list, prediction_time).add_suffix(k_name))
#     df_res = pd.concat(all_df_k, axis=1).loc[['ARIMA', 'Prophet', 'NeuralProphet', 'Event NeuralProphet', 'GAN - Event CNN', 'LSTM', 'Event LSTM', 'Weighted Event LSTM', 'GAN - Event LSTM']]
#     return df_res
#
#
# def plot_k_graph(total_pred, peaks_data, final_list, leaf_name, start_pred_list, prediction_time, metric='MAE'):
#     all_df_k = []
#     for k in range(1, 51):
#         all_df_k.append(get_metrics_at_k(k, total_pred, peaks_data, final_list, leaf_name, start_pred_list, prediction_time, metric).add_suffix('@' + str(k)))
#     df_res = pd.concat(all_df_k, axis=1).loc[['ARIMA', 'Prophet', 'NeuralProphet', 'Event NeuralProphet', 'GAN - Event CNN', 'LSTM', 'Event LSTM', 'Weighted Event LSTM', 'GAN - Event LSTM']]
#     return df_res
#
#
# def get_paper_results(combine_res):
#     combine_res = combine_res.rename(index={'NeuralProphet': 'Neural Prophet',
#                                             'Event NeuralProphet': 'Event Neural Prophet',
#                                             'GAN - Event LSTM': 'GAN-Event LSTM',
#                                             'GAN - Event CNN': 'GAN-Event CNN'})
#     paper_res = combine_res
#     paper_res.columns = paper_res.columns.get_level_values(0).map(lambda h: '\\textbf{' + h.replace('%', '\\%') + '}')
#     paper_res.index = '\\textbf{' + paper_res.index + '}'
#
#     agg_dict = {
#         '\\textbf{MAE@5}': 'min',
#         '\\textbf{wMAPE@5}': 'min',
#
#         '\\textbf{MAE@10}': 'min',
#         '\\textbf{wMAPE@10}': 'min',
#
#         '\\textbf{MAE@20}': 'min',
#         '\\textbf{wMAPE@20}': 'min',
#
#         '\\textbf{MAE@25}': 'min',
#         '\\textbf{wMAPE@25}': 'min',
#     }
#
#     def col_formatter(col):
#         if agg_dict[col] == 'max':
#             m = paper_res[col].max()
#         else:
#             m = paper_res[col].min()
#
#         def _f(x):
#             v = x
#             x = "%.3f" % round(x, 3) if 'MAE' not in col else str(v)
#             return '\\textbf{' + x + '}' if v == m else x
#
#         return _f
#
#     df_latex = (
#         paper_res.reset_index().to_latex(
#             index=False,
#             bold_rows=True,
#             escape=False,
#             formatters=[None] + [col_formatter(col) for col in paper_res.columns]
#         )
#     )
#
#     return combine_res, paper_res, df_latex


