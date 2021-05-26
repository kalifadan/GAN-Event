import torch
import pytorch_lightning as pl
from src.models.EventsDataset import EventsDataModule
from src.models.EventModel import EventModel
from src.config import LOG_DIR, SEED

pl.seed_everything(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

hparams = {
    'dataset_name': "world_events_dataset_from_1980",
    # 'events_path': '',
    'log_path': LOG_DIR,
    'log_every_n_steps': 1,

    'gpus': '2',
    'batch_size': 32,

    'wikipedia_only': True,
    'filter_days_with_one_event': True,

    'training_stage2': True,
    'start_test_date': '2020-01-01',

    'single_events_percent': 0,
    'window_size': 2,

    'num_layers': 2,
    'nhead': 4,
    'activation': 'leaky_relu',
    'mask_percent': 0.25,

    'loss_type': 'HausdorffLoss',       # choose from: ['HausdorffLoss', 'L1', 'L2', 'CosineEmbedding']
    'hausdorff_type': 'Cosine',         # choose from: ['L1', 'L2', 'Cosine'] or provide a dist function

    'g_lmb': 10,
    'd_lmb': 1,

    'lr_gen': 1e-4,
    'weight_decay_gen': 0.001,

    'lr_dis': 1e-4,
    'weight_decay_dis': 0.001,

    'epochs': 100,
    'saving_results': True,
    'results_prefix': 'GAN-Embeddings'
}


if __name__ == '__main__':
    dm = EventsDataModule(hparams=hparams)
    dm.prepare_data()
    dm.setup()

    model = EventModel(hparams=hparams)
    model.fit(datamodule=dm)
    model.test(datamodule=dm)

