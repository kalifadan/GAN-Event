import numpy as np
from darts.models.tcn_model import TCNModel
from darts.utils.data.shifted_dataset import ShiftedDataset


class MyShiftedDataset(ShiftedDataset):
    """
     Imported from darts library and edited to predict only the ts and not the events in each prediction step.
    """
    def __getitem__(self, idx: int):
        # determine the index of the time series.
        ts_idx = idx // self.max_samples_per_ts
        ts_target = self.target_series[ts_idx].values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(ts_target) - self.length - self.shift + 1

        # Determine the index of the end of the output, starting from the end.
        # It is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        end_of_output_idx = (idx - (ts_idx * self.max_samples_per_ts)) % n_samples_in_ts

        # select forecast point and target period, using the previously computed indexes
        if end_of_output_idx == 0:
            # we need this case because "-0" is not supported as an indexing bound
            output_series = ts_target[-self.length:]
            output_series = np.expand_dims(np.asarray(output_series[:, 0]), axis=1)   # line changed
        else:
            output_series = ts_target[-(self.length + end_of_output_idx):-end_of_output_idx]
            output_series = np.expand_dims(np.asarray(output_series[:, 0]), axis=1)   # line changed

        input_series = ts_target[-(self.length + end_of_output_idx + self.shift):-(end_of_output_idx + self.shift)]

        input_covariate = None
        if self.covariates is not None:
            ts_covariate = self.covariates[ts_idx].values(copy=False)
            input_covariate = ts_covariate[-(self.length + end_of_output_idx + self.shift):-(end_of_output_idx + self.shift)]

        return input_series, output_series, input_covariate


class EventTCNModel(TCNModel):
    def _build_train_dataset(self, target, covariates):
        return MyShiftedDataset(target_series=target,
                              covariates=covariates,
                              length=self.input_chunk_length,
                              shift=1)