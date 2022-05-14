import os
from utils.util import *
from train_test import *

# Options: demo, small, large
MIND_type = 'small'
data_path = "./data/"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
knowledge_graph_file = os.path.join(data_path, 'kg/wikidata-graph', r'wikidata-graph.tsv')
entity_embedding_file = os.path.join(data_path, 'kg/wikidata-graph', r'entity2vecd100.vec')
relation_embedding_file = os.path.join(data_path, 'kg/wikidata-graph', r'relation2vecd100.vec')

mind_url, mind_train_dataset, mind_dev_dataset, _ = get_mind_data_set(MIND_type)
print(mind_url)
kg_url = "https://kredkg.blob.core.windows.net/wikidatakg/"

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    
if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)

if not os.path.exists(knowledge_graph_file):
    download_deeprec_resources(kg_url, \
                               os.path.join(data_path, 'kg'), "kg.zip")


import sys
import os
sys.path.append('')

import argparse
from parse_config import ConfigParser

parser = argparse.ArgumentParser(description='KRED')


parser.add_argument('-c', '--config', default="./config.json", type=str,
                    help='config file path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-d', '--device', default=None, type=str,
                    help='indices of GPUs to enable (default: all)')

config = ConfigParser.from_args(parser)


epochs = 20
batch_size = 64
train_type = "single_task"
task = "user2item" # task should be within: user2item, item2item, vert_classify, pop_predict

config['trainer']['epochs'] = epochs
config['data_loader']['batch_size'] = batch_size
config['trainer']['training_type'] = train_type
config['trainer']['task'] = task

data = load_data_mind(config)

if train_type == "single_task":
    single_task_training(config, data)
else:
    multi_task_training(config, data)


test_data = data[-1]
testing(test_data, config)

