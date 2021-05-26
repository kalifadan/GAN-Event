import pandas as pd
from src.run import hparams
from src.config import DATA_DIR

EMB_DIM = 100

RESULTS_PREFIX = hparams['results_prefix']
EMBEDDINGS_TRAIN_GAN_PATH = DATA_DIR + "/gan_embeddings/" + RESULTS_PREFIX + "_train_gan_embeddings.pkl"
EMBEDDINGS_TEST_GAN_PATH = DATA_DIR + "/gan_embeddings/" + RESULTS_PREFIX + "_test_gan_embeddings.pkl"


def create_embeddings_df(embeddings_df, train_df, test_df, categ_data, limit_size=1):
    gen_data = categ_data.merge(embeddings_df, on='date', how='left')
    gen_data = gen_data.fillna(method='ffill', limit=limit_size)      # look on limit size days before for nan values
    gen_data = gen_data.fillna(-1)
    gen_data['embeddings'] = gen_data['embeddings'].apply(lambda x: [0.0] * EMB_DIM if x == -1 else x)
    for ii in range(EMB_DIM):
        col = 'emb_' + str(ii)
        gen_data[col] = gen_data['embeddings'].apply(lambda x: x[ii])
    gen_data.index = gen_data['date']
    gen_data = gen_data.drop(columns=['embeddings', 'date'])
    X_gen = gen_data.drop(columns=['Quantity'])
    X_gen.index = X_gen.index.astype('datetime64[ns]')
    train_set = train_df.merge(X_gen, on='date')
    test_set = test_df.merge(X_gen, on='date')
    return train_set, test_set


def get_gan_embeddings(train_df, test_df, categ_data, window_size=2):
    train_gan_df = pd.read_pickle(EMBEDDINGS_TRAIN_GAN_PATH)
    test_gan_df = pd.read_pickle(EMBEDDINGS_TEST_GAN_PATH)
    embeddings_df = pd.concat([train_gan_df, test_gan_df], ignore_index=True)
    train_set, test_set = create_embeddings_df(embeddings_df, train_df, test_df, categ_data, limit_size=(window_size-1))
    return train_set, test_set


