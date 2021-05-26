import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from src.models.EventsEmbeddings import EventsEmbeddings
from src.models.Transformers import MyTransformerEncoderLayer, MyTransformerEncoder
from src.models.HausdorffLoss import HausdorffLoss

WIKI_VEC_SIZE = 100


class EventsModule(pl.LightningModule):
    def __init__(self, hparams, emb_dict):
        super(EventsModule, self).__init__()
        self.hparams = hparams
        self.mask_percent = hparams['mask_percent'] if 'mask_percent' in hparams else 0.3
        self.nhead = hparams['nhead'] if 'nhead' in hparams else 4
        self.num_layers = hparams['num_layers'] if 'num_layers' in hparams else 6
        self.wd_gen = hparams['weight_decay_gen'] if 'weight_decay_gen' in hparams else 0
        self.lr_gen = hparams['lr_gen'] if 'lr_gen' in hparams else 0.01
        self.embedding_model = EventsEmbeddings(emb_dict, hparams=hparams)
        self.out_dim = self.embedding_model.out_dim

        self.encoder_layer = MyTransformerEncoderLayer(d_model=self.out_dim, nhead=self.nhead)
        self.transformer_encoder = MyTransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.masked_event = nn.Parameter(torch.randn(WIKI_VEC_SIZE, requires_grad=True, dtype=torch.float, device=self.device))
        self.emtpy_event = nn.Parameter(torch.randn(self.out_dim, requires_grad=True, dtype=torch.float, device=self.device))

        self.loss_type = hparams['loss_type'] if 'loss_type' in hparams else 'L1'
        self.hausdorff_type = hparams['hausdorff_type'] if 'hausdorff_type' in hparams else 'L2'

        if self.loss_type == 'L1':
            self.loss_func = nn.L1Loss()
        elif self.loss_type == 'L2':
            self.loss_func = nn.MSELoss()
        elif self.loss_type == 'CosineEmbedding':
            self.loss_func = nn.CosineEmbeddingLoss()
        elif self.loss_type == 'HausdorffLoss':
            self.loss_func = HausdorffLoss(self.hausdorff_type)
        else:
            raise Exception('No such loss', self.loss_type)

    def gen_loss(self, pred, target):
        if self.loss_type == 'L1' or self.loss_type == 'L2' or self.loss_type == 'HausdorffLoss':
            return self.loss_func(pred, target)
        elif self.loss_type == 'CosineEmbedding':
            return self.loss_func(pred, target, torch.ones(pred.size(0), device=self.device))
        else:
            raise Exception('No such loss', self.loss_type)

    def get_masked_embedding(self):
        zero_tensor = torch.tensor([0], device=self.device).unsqueeze(0)
        masked_dict = {
            'country': zero_tensor,
            'High-Category': zero_tensor,
            'Category': zero_tensor,
            'embeddings': self.masked_event.unsqueeze(0).unsqueeze(0)
        }
        return self.embedding_model(masked_dict).squeeze().squeeze()

    def forward(self, x, test_mode=False):
        # ----------------------- EMBEDDING LAYER -----------------------
        src, lengths = x
        N, S, E = src.shape

        # -------------------------- MASKING -------------------------
        # Mask (BoolTensor): positions with 'True' are not allowed to attend while 'False' values will be unchanged.
        src_padding_mask = torch.zeros((N, S), dtype=torch.bool, device=self.device)
        mask_indices = torch.zeros((N, S), dtype=torch.bool, device=self.device)
        empty_indices = torch.zeros((N, S), dtype=torch.bool, device=self.device)
        masked_embedding = self.get_masked_embedding()

        for row_idx, row_len in enumerate(lengths):
            src_padding_mask[row_idx, row_len:] = True  # true for the padding indices

            if not test_mode:
                if row_len > 1:
                    num_events_masked = np.ceil(row_len.cpu() * self.mask_percent)  # chose mask % events to be masked
                    events_masked_indices = torch.tensor(np.random.choice(row_len.cpu(), int(num_events_masked),
                                                                          replace=False), device=self.device)
                    mask_indices[row_idx, events_masked_indices] = True
                else:
                    assert S > 1, "seq len must be at least 2 - in case of single event, add empty event and mask it"
                    src_padding_mask[row_idx, row_len] = False
                    mask_indices[row_idx, row_len] = True
                    empty_indices[row_idx, row_len] = True

        if not test_mode:
            src[mask_indices, :] = masked_embedding  # mask the chosen events
            src[empty_indices, :] = self.emtpy_event  # replace empty mask with the empty event

        # -------------------------- TRANSFORMER -------------------------
        src = src.transpose(0, 1)  # input dim of (seq len, batch size, embedding dim)
        out, weights_matrix_list = self.transformer_encoder(src=src, src_key_padding_mask=src_padding_mask)
        out = out.transpose(0, 1)

        if not test_mode:
            return out, mask_indices
        else:
            return out, weights_matrix_list

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_gen, weight_decay=self.wd_gen)
        return [optimizer]







