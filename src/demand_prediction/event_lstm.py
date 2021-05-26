import numpy as np
from darts.models.rnn_model import RNNModel
from darts.utils.data.sequential_dataset import SequentialDataset


class MySequentialDataset(SequentialDataset):
    """
     Imported from darts library and edited to predict only the ts and not the events in each prediction step.
    """
    def __getitem__(self, idx: int):
        # determine the index of the time series.
        ts_idx = idx // self.max_samples_per_ts
        ts_target = self.target_series[ts_idx].values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(ts_target) - self.input_chunk_length - self.output_chunk_length + 1

        # Determine the index of the forecasting point.
        # It is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        lh_idx = (idx - (ts_idx * self.max_samples_per_ts)) % n_samples_in_ts

        # The time series index of our forecasting point (indexed from the end of the series):
        forecast_point_idx = self.output_chunk_length + lh_idx

        # select input and outputs, using the previously computed indexes
        input_target = ts_target[-(forecast_point_idx + self.input_chunk_length):-forecast_point_idx]
        if forecast_point_idx == self.output_chunk_length:
            # we need this case because "-0" is not supported as an indexing bound
            output_target = ts_target[-forecast_point_idx:]
            output_target = np.asarray([output_target[:, 0]])  # line changed, to predict only the ts (not the events)
        else:
            output_target = ts_target[-forecast_point_idx:-forecast_point_idx + self.output_chunk_length]
            output_target = np.asarray([output_target[:, 0]])   # line changed, to predict only the ts (not the events)

        input_covariate = None
        if self.covariates is not None:
            ts_covariate = self.covariates[ts_idx].values(copy=False)
            input_covariate = ts_covariate[-(forecast_point_idx + self.input_chunk_length):-forecast_point_idx]

        return input_target, output_target, input_covariate


class EventRNNModel(RNNModel):
    def _build_train_dataset(self, target, covariates):
        return MySequentialDataset(target_series=target,
                                 covariates=covariates,
                                 input_chunk_length=self.input_chunk_length,
                                 output_chunk_length=self.output_chunk_length)