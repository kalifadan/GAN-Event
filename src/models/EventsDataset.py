import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch
import src.config as proj_config


class EventDataset(Dataset):
    @staticmethod
    def get_embeddings_vec(events, feature_name):
        embeddings_vec = events[feature_name].values
        embeddings = [torch.tensor(vec, dtype=torch.float64) for vec in embeddings_vec]
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings

    @staticmethod
    def check_diff_in_days(day1, day2, diff):
        dt1 = pd.to_datetime(day1, format='%Y/%m/%d')
        dt2 = pd.to_datetime(day2, format='%Y/%m/%d')
        return abs((dt1 - dt2).days) <= diff

    def __init__(self, dataset, dates, embeddings_dict, window_size):
        self.dataset = dataset
        self.dates = dates
        self.embeddings_dict = embeddings_dict
        self.window_size = window_size

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, index):
        if self.window_size > 1:        # take all events in current date until window size before (including current)
            date = self.dates[index]
            start_idx = max(0, index - self.window_size + 1)
            opt_dates = self.dates[start_idx: index + 1]
            dates = [d for d in opt_dates if EventDataset.check_diff_in_days(date, d, self.window_size)]
            events_cur_date = self.dataset[self.dataset['date'].isin(dates)]
        else:
            date = self.dates[index]
            events_cur_date = self.dataset[self.dataset['date'] == date]

        res = {'embeddings': self.get_embeddings_vec(events_cur_date, 'embedding')}
        for feature_name in self.embeddings_dict.keys():
            res[feature_name] = torch.tensor(events_cur_date[feature_name].values, dtype=torch.int64)

        res['wiki_name'] = events_cur_date['wiki_name'].values  # saving events name for the feature selection later
        return res


class EventsDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.dataset_name = self.hparams['dataset_name']
        self.batch_size = self.hparams['batch_size'] if 'batch_size' in hparams else 32
        self.num_workers = 0   # must be zero (only a single worker)
        self.embeddings_features = ['country', 'High-Category', 'Category']
        self.embeddings_dict = {}
        self.label_encoder = LabelEncoder()
        self.train, self.val, self.test = None, None, None
        self.dataset, self.events_dates, self.invalid_dates, self.embedding_train = None, None, None, None
        self.embedding_train_dates, self.train_dates, self.val_dates, self.test_dates = None, None, None, None
        self.embedding_sizes = {'country': 20, 'High-Category': 3, 'Category': 5}
        self.filter_single_days = self.hparams['filter_days_with_one_event'] if 'filter_days_with_one_event' in hparams else False
        self.filter_percent = self.hparams['single_events_percent'] if 'single_events_percent' in hparams else 0
        self.window_size = self.hparams['window_size'] if 'window_size' in hparams else 1
        self.start_test_date = self.hparams['start_test_date'] if 'start_test_date' in hparams else '2020-01-01'
        self.training_stage2 = self.hparams['training_stage2'] if 'training_stage2' in hparams else True

    def prepare_data(self):
        if 'events_path' not in self.hparams:
            file_path = os.path.join(proj_config.EVENTS_DATASET_PATH, f'{self.dataset_name}.pkl')
        else:
            file_path = os.path.join(self.hparams['events_path'], f'{self.dataset_name}.pkl')

        if os.path.exists(file_path):
            self.dataset = pd.read_pickle(file_path).reset_index(drop=True)
            self.dataset = self.dataset.sort_values(by=['date']).reset_index(drop=True)
            dates = list(self.dataset['date'].values)
            self.events_dates = sorted(set(dates), key=dates.index)
        else:
            raise RuntimeError('Failed to find dataset')

        if self.filter_single_days:
            single_day_events = []
            for date in self.events_dates:
                window_dates = pd.date_range(end=date, periods=self.window_size, closed='right').tolist()
                window_dates = [str(d).split()[0] for d in window_dates]
                cur_events = self.dataset[self.dataset['date'].isin(window_dates)]
                if len(cur_events) <= 1:
                    single_day_events.append(date)
            invalid_days_amount = int(len(single_day_events) * (1 - self.filter_percent))
            self.invalid_dates = np.random.choice(single_day_events, invalid_days_amount, replace=False)

        print("Total Events In Dataset:", len(self.dataset))

        for feature in self.embeddings_features:
            unique_values = self.dataset[feature].nunique()
            self.dataset[feature] = self.label_encoder.fit_transform(self.dataset[feature]) + 1   # for padding
            self.embeddings_dict[feature] = {
                'num_embeddings': unique_values + 1,   # for padding
                'embedding_dim': self.embedding_sizes[feature],
                'labels_dict': ['Padding'] + list(self.label_encoder.classes_)
            }

    def setup(self, stage=None):
        dates_indices = list(range(len(self.events_dates)))
        print("Total Samples In Dataset:", len(self.events_dates))
        print("Window Size:", self.window_size)

        if self.training_stage2:
            print("Training Stage 2 - Without Validation Set")
            train_indices = [idx for idx, date in enumerate(self.events_dates) if date < self.start_test_date]
            self.train_dates = [self.events_dates[ii] for ii in train_indices]
            self.embedding_train = EventDataset(self.dataset, self.train_dates, self.embeddings_dict, self.window_size)
            self.embedding_train_dates = self.train_dates
            self.train_dates = [date for date in self.train_dates if date not in self.invalid_dates]  # drop single days
            self.train = EventDataset(self.dataset, self.train_dates, self.embeddings_dict, self.window_size)

            self.val = None

            test_indices = [idx for idx, date in enumerate(self.events_dates) if date >= self.start_test_date]
            self.test_dates = [self.events_dates[ii] for ii in test_indices]
            self.test = EventDataset(self.dataset, self.test_dates, self.embeddings_dict, self.window_size)

        else:
            print("Training Stage 1 - With Validation Set")
            test_start_idx = 0
            for ii, date in enumerate(self.events_dates):
                if date >= self.start_test_date:
                    test_start_idx = ii
                    break

            test_indices = dates_indices[test_start_idx:]
            self.test_dates = [self.events_dates[ii] for ii in test_indices]
            self.test = EventDataset(self.dataset, self.test_dates, self.embeddings_dict, self.window_size)

            train_indices = dates_indices[:int(test_start_idx * 0.8)]
            self.train_dates = [self.events_dates[ii] for ii in train_indices]
            self.embedding_train = EventDataset(self.dataset, self.train_dates, self.embeddings_dict, self.window_size)
            self.embedding_train_dates = self.train_dates
            self.train_dates = [date for date in self.train_dates if date not in self.invalid_dates]  # drop single days
            self.train = EventDataset(self.dataset, self.train_dates, self.embeddings_dict, self.window_size)

            val_indices = dates_indices[int(test_start_idx * 0.8): test_start_idx]
            self.val_dates = [self.events_dates[ii] for ii in val_indices]
            self.val = EventDataset(self.dataset, self.val_dates, self.embeddings_dict, self.window_size)

    def collate_fn(self, batch):
        batch_dict, lengths = {}, None
        keys = list(self.embeddings_dict.keys()) + ['embeddings']   # Tensor containing the number of events per sample

        for feature_name in keys:
            lengths = torch.tensor([t[feature_name].shape[0] for t in batch])
            batch_feature = [t[feature_name] for t in batch]
            batch_dict[feature_name] = torch.nn.utils.rnn.pad_sequence(batch_feature, batch_first=True, padding_value=0)

        if lengths.sum() == lengths.size(0):    # for batch with seq_len == 1 add extra padding
            for f in keys:
                val = batch_dict[f]
                padding = torch.zeros_like(val)
                batch_dict[f] = torch.cat((val, padding), dim=1)

        batch_dict['wiki_name'] = [t['wiki_name'] for t in batch]
        return batch_dict, lengths

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, drop_last=False)

    def embedding_train_dataloader(self):
        return DataLoader(self.embedding_train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn, drop_last=False)

    def get_dates(self):
        return self.embedding_train_dates, self.train_dates, self.val_dates, self.test_dates




