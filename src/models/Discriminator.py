import torch
from torch import nn
import pytorch_lightning as pl


class DiscriminatorModel(pl.LightningModule):
    def __init__(self, out_dim, hparams):
        super().__init__()
        self.out_dim = out_dim
        self.out_activation = nn.Sigmoid()
        self.activation_type = hparams['activation'] if 'activation' in hparams else 'relu'
        self.activation = nn.ReLU() if self.activation_type == 'relu' else nn.LeakyReLU()
        self.nhead = hparams['nhead'] if 'nhead' in hparams else 4
        self.num_layers = hparams['num_layers'] if 'num_layers' in hparams else 6
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.out_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.linear1 = nn.Linear(self.out_dim, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 1)

    def forward(self, x):
        src, lengths = x
        N, S, E = src.shape

        # -------------------------- MASKING -------------------------
        src_padding_mask = torch.zeros((N, S), dtype=torch.bool, device=self.device)
        for row_idx, row_len in enumerate(lengths):
            src_padding_mask[row_idx, row_len:] = True  # true for the padding indices

        # -------------------------- TRANSFORMER -------------------------
        out = self.transformer_encoder(src=src.transpose(0, 1), src_key_padding_mask=src_padding_mask).transpose(0, 1)

        # sum the second dim without the padding
        new_out = torch.zeros((out.shape[0], out.shape[2]), dtype=torch.float, device=self.device)
        for row_idx, row_len in enumerate(lengths):
            new_out[row_idx] = out[row_idx][:row_len].transpose(0, 1).sum(dim=1) / row_len
        out = new_out

        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.linear3(out)
        out = self.out_activation(out)

        return out

