from datetime import datetime
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.models.EventsModule import EventsModule
from src.models.AdversarialEventsModule import AdversarialEventsModule


class EventModel:
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        name = f"Model Results"
        self.logger = WandbLogger(name=name, save_dir=hparams['log_path'],
                                  version=datetime.now().strftime('%y%m%d_%H%M%S.%f'),
                                  project='world_events_project', entity='world-events-project', config=hparams)
        self.events_model, self.trainer, self.model = None, None, None
        self.training_stage2 = self.hparams['training_stage2'] if 'training_stage2' in hparams else True

    def fit(self, datamodule):
        self.events_model = EventsModule(self.hparams, emb_dict=datamodule.embeddings_dict)
        self.model = AdversarialEventsModule(self.hparams, datamodule.embeddings_dict, self.events_model)

        if self.training_stage2:
            monitor_checkpoint = 'generator/train_loss'
            monitor_early_stop = 'discriminator/train_discriminator_loss'
            # check_val_epoch need to be bigger than epochs no. since we don't have a validation set
            check_val_epoch = self.hparams['epochs'] + 100
            num_sanity_val_steps = 0

        else:
            monitor_checkpoint = 'generator/val_loss'
            monitor_early_stop = 'discriminator/val_discriminator_loss'
            check_val_epoch = 1
            num_sanity_val_steps = 2

        checkpoint_callback = ModelCheckpoint(save_top_k=1, verbose=True, save_last=True, monitor=monitor_checkpoint,
                                              mode='min', filename='best_model')
        early_stooping = EarlyStopping(monitor=monitor_early_stop, patience=25, mode='min')

        self.trainer = pl.Trainer(gpus=self.hparams['gpus'], max_epochs=self.hparams['epochs'],
                                  logger=self.logger, log_every_n_steps=self.hparams['log_every_n_steps'],
                                  callbacks=[early_stooping, checkpoint_callback], deterministic=True,
                                  num_sanity_val_steps=num_sanity_val_steps, check_val_every_n_epoch=check_val_epoch)

        self.trainer.fit(self.model, datamodule=datamodule)

        save_model_path = f"{self.hparams['results_prefix'] + 'Final_Model'}.ckpt"
        self.trainer.save_checkpoint(os.path.join(self.hparams['log_path'], save_model_path))

    def test(self, datamodule):
        self.trainer.test(datamodule=datamodule)
        self.trainer.logger.experiment.log({}, commit=True)

