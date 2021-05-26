import os
from os.path import join

SEED = 0

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = join(ROOT_DIR, '../data')
CACHE_DIR = join(ROOT_DIR, '../cache')
EVENTS_DATASET_DIR = join(ROOT_DIR, '../data/datasets/events')
ECOMMERCE_DATASET_DIR = join(ROOT_DIR, '../data/datasets/ecommerce')
EVENTS_DATASET_PATH = EVENTS_DATASET_DIR + "/"
LOG_DIR = join(ROOT_DIR, '../log')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVENTS_DATASET_DIR, exist_ok=True)
os.makedirs(ECOMMERCE_DATASET_DIR, exist_ok=True)
