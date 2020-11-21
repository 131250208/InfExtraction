'''
Train the model
'''

# load data
# Split the data (by max sequence length in the config file)
# Index the data
# Put the data into a Dataloader (from torch.utils.data import DataLoader, Dataset)
# Initialize the model and metrics
# Train

from InfExtraction.components.preprocess import Preprocessor, MyDataset
from InfExtraction.components.taggers import HandshakingTagger4EE
from InfExtraction.work_flows import settings_train as settings
from InfExtraction.work_flows.utils import DefaultLogger
from InfExtraction.components.models import TPLinkerPlus
import os
import torch
import wandb
import json
from pprint import pprint
from torch.utils.data import DataLoader

# settings
exp_name = settings.exp_name
data_in_dir = settings.data_in_dir
train_data_path = os.path.join(data_in_dir, exp_name, settings.train_data)
valid_data_path = os.path.join(data_in_dir, exp_name, settings.valid_data)
dicts_path = os.path.join(data_in_dir, exp_name, settings.dicts)
statistics_path = os.path.join(data_in_dir, exp_name, settings.statistics)

use_wandb = settings.use_wandb
run_name = settings.run_name
config2log = settings.config_to_log
# logger settings
log_path = settings.log_path
run_id = settings.run_id
path_to_save_model = settings.path_to_save_model
# training settings
device_num = settings.device_num
task_type = settings.task_type
language = settings.language
use_bert = settings.use_bert
seed = settings.seed
batch_size = settings.batch_size
epochs = settings.epochs
log_interval = settings.log_interval
max_seq_len = settings.max_seq_len
sliding_len = settings.sliding_len
scheduler = settings.scheduler
ghm = settings.ghm
tok_pair_sample_rate = settings.tok_pair_sample_rate
# model
model_settings = settings.model_settings

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed) # pytorch random seed
torch.backends.cudnn.deterministic = True # for reproductivity

# reset settings from args

# logger
if use_wandb:
    # init wandb
    wandb.init(project=exp_name, name=run_name, config=config2log)
    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    logger = DefaultLogger(log_path,
                           exp_name,
                           run_name,
                           run_id, config2log)
    model_state_dict_dir = path_to_save_model
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

# load data
train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
statistics = json.load(open(statistics_path, "r", encoding="utf-8"))
dicts = json.load(open(dicts_path, "r", encoding="utf-8"))

# splitting
if use_bert:
    max_seq_len_statistics = statistics["max_subword_seq_length"]
    feature_list_key = "subword_level_features"
else:
    max_seq_len_statistics = statistics["max_word_seq_length"]
    feature_list_key = "word_level_features"
max_seq_len = min(max_seq_len, max_seq_len_statistics)
split_train_data = Preprocessor.split_into_short_samples(train_data, max_seq_len, sliding_len, "train",
                                                         wordpieces_prefix=model_settings["wordpieces_prefix"],
                                                         feature_list_key=feature_list_key)
split_valid_data = Preprocessor.split_into_short_samples(valid_data, max_seq_len_statistics, sliding_len, "valid",
                                                         wordpieces_prefix=model_settings["wordpieces_prefix"],
                                                         feature_list_key=feature_list_key)
sample_id2dismatched = Preprocessor.check_splits(split_train_data)
pprint(sample_id2dismatched)
sample_id2dismatched = Preprocessor.check_splits(split_valid_data)
pprint(sample_id2dismatched)

# indexing
key2dict = {
    "char_list": dicts["char2id"],
    "word_list": dicts["word2id"],
    "pos_tag_list": dicts["pos_tag2id"],
    "ner_tag_list": dicts["ner_tag2id"],
    "dependency_list": dicts["deprel_type2id"],
}
pretrained_model_padding = model_settings["pretrained_model_padding"] if "pretrained_model_padding" in model_settings else 0
indexed_train_data = Preprocessor.index_features(split_train_data, key2dict, max_seq_len, pretrained_model_padding)
indexed_valid_data = Preprocessor.index_features(split_valid_data, key2dict, max_seq_len_statistics, pretrained_model_padding)

# tagging
tagger4train_data = HandshakingTagger4EE(dicts["rel_type2id"], dicts["ent_type2id"], max_seq_len)
tagger4valid_data = HandshakingTagger4EE(dicts["rel_type2id"], dicts["ent_type2id"], max_seq_len_statistics)

indexed_train_data = tagger4train_data.tag(indexed_train_data)
indexed_valid_data = tagger4valid_data.tag(indexed_valid_data)

# dataset
train_dataloader = DataLoader(MyDataset(indexed_train_data),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=6,
                              drop_last=False,
                              collate_fn=TPLinkerPlus.generate_batch,
                             )
valid_dataloader = DataLoader(MyDataset(indexed_valid_data),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=6,
                              drop_last=False,
                              collate_fn=TPLinkerPlus.generate_batch,
                             )

