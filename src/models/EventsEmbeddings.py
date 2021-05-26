import torch.nn as nn

WIKI_VEC_SIZE = 100


class EventsEmbeddings(nn.Module):
    def __init__(self, embeddings_dict, hparams):
        super().__init__()
        self.wiki_only = hparams['wikipedia_only'] if 'wikipedia_only' in hparams else True
        self.embeddings = nn.ModuleList([nn.Embedding(embeddings_dict[feature]['num_embeddings'],
                            embeddings_dict[feature]['embedding_dim'], padding_idx=0) for feature in embeddings_dict])
        self.embeddings_features = embeddings_dict.keys()
        self.out_dim = sum(e.embedding_dim for e in self.embeddings) + WIKI_VEC_SIZE if not self.wiki_only else WIKI_VEC_SIZE
        self.emb_linear = nn.Linear(self.out_dim, self.out_dim)
        self.layer_norm = nn.LayerNorm(self.out_dim)
        self.activation_type = hparams['activation'] if 'activation' in hparams else 'relu'
        self.activation = nn.ReLU() if self.activation_type == 'relu' else nn.LeakyReLU()

    def forward(self, batch_dict):
        if self.wiki_only:
            x = batch_dict['embeddings'].float()
            return x
        else:
            raise RuntimeError("Not supported yet.")










