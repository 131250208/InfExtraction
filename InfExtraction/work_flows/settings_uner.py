import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
import torch
import random
import numpy as np
from datetime import date
import time
from InfExtraction.modules.utils import load_data, get_oss_client
from transformers import BertTokenizer

seed = 2333
enable_bm = True


def set_seed():
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def enable_benchmark():
    torch.backends.cudnn.enabled = enable_bm
    torch.backends.cudnn.benchmark = enable_bm
    torch.backends.cudnn.deterministic = True


set_seed()
enable_benchmark()

import string
import json
import copy
import re
from glob import glob

# Frequent changes
exp_name = "genia"
language = "en"
do_lower_case = False
word_tokenizer_type = "white"

task_type = "ner"
model_name = "UNER"
tagger_name = "UNERTagger"
# run_name = "{}+{}+{}".format(task_type, re.sub("[^A-Z]", "", model_name), re.sub("[^A-Z]", "", tagger_name))
run_name = task_type
pretrained_model_name = "biobert_large_v1.1_squad"  # biobert_large， bert-large-cased
pretrained_emb_name = "bio_embedding_extrinsic.bin"
use_wandb = True
note = ""
epochs = 100
lr = 1e-5  # 5e-5, 1e-4
check_tagging_n_decoding = True
combine = False  # combine splits
scheduler = "CAWR"
model_bag_size = 0
metric_pattern2save = None  # if none, save best models on all metrics

batch_size_train = 12
batch_size_valid = 12
batch_size_test = 12

max_seq_len_train = 100
max_seq_len_valid = 100
max_seq_len_test = 100

sliding_len_train = 30
sliding_len_valid = 30
sliding_len_test = 30

# >>>>>>>>>>>>>>>>> features >>>>>>>>>>>>>>>>>>>
token_level = "subword"  # token is word or subword

# to do an ablation study, you can ablate components by setting it to False
pos_tag_emb = False
ner_tag_emb = False
char_encoder = False
dep_gcn = False

word_encoder = True
subwd_encoder = True
use_attns4rel = True

# data
data_in_dir = "inside:s3://wycheng_b1/data/info_extr/preprocessed_data"
data_out_dir = "../../data/res_data"

train_data_path = "/".join([data_in_dir, exp_name, "train_data.json"])
valid_data_path = "/".join([data_in_dir, exp_name, "test_data.json"])

test_data_path_list = []
# cluster = 'inside'
# files = get_oss_client().get_file_iterator("{}/{}/".format(data_in_dir, exp_name))
# for p, k in files:
#     if "test" in p:
#         path = '{0}:s3://{1}'.format(cluster, p)
#         test_data_path_list.append(path)

dicts = "dicts.json"
statistics = "statistics.json"
statistics_path = "{}/{}/{}".format(data_in_dir, exp_name, statistics)
dicts_path = "{}/{}/{}".format(data_in_dir, exp_name, dicts)
statistics = load_data(statistics_path)
dicts = load_data(dicts_path)

# for preprocessing
key_map = {
    "char2id": "char_list",
    "word2id": "word_list",
    # "bert_dict": "subword_list",
    "pos_tag2id": "pos_tag_list",
    "ner_tag2id": "ner_tag_list",
    "deprel_type2id": "dependency_list",
}
pretrained_model_path = "../../data/pretrained_models/{}".format(pretrained_model_name)
key2dict = {
    "subword_list": BertTokenizer.from_pretrained(pretrained_model_path).get_vocab()
}
for key, val in dicts.items():
    if key in key_map:
        key2dict[key_map[key]] = val

# additional preprocessing
addtional_preprocessing_config = {
    "add_default_entity_type": False,
    "classify_entities_by_relation": False
}

# tagger config
tagger_config = {
    "ent_type2desc": "{}/{}/{}".format(data_in_dir, exp_name, "ent_type2desc.json"),
    "type_num": 7,
    "classify_entities_by_relation": addtional_preprocessing_config["classify_entities_by_relation"],
    "add_h2t_n_t2h_links": False,
    "add_o2s_links": False,
    "language": language
}

# optimizers and schedulers
optimizer_config = {
    "class_name": "Adam",
    "parameters": {
    }
}

scheduler_dict = {
    "CAWR": {
        # CosineAnnealingWarmRestarts
        "name": "CAWR",
        "T_mult": 1,
        "rewarm_epochs": 10,
    },
    "StepLR": {
        "name": "StepLR",
        "decay_rate": 0.999,
        "decay_steps": 100,
    },
}

# logger
log_interval = 10
# for default logger
random.seed(time.time())
default_run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
random.seed(seed)
default_log_path = "./default_log_dir/default.log"
default_dir_to_save_model = "./default_log_dir/run-{}_{}-{}".format(date.today().strftime("%Y%m%d"),
                                                                    str(time.time())[:6],
                                                                    default_run_id)
# trainer config
trainer_config = {
    "run_name": run_name,
    "run_id": default_run_id,
    "exp_name": exp_name,
    "scheduler_config": scheduler_dict[scheduler],
    "log_interval": log_interval,
}

# pretrianed model state path, for continuous training
model_state_dict_path = None

# for inference and evaluation
model_dir_for_test = "./wandb"  # "./default_log_dir" or "./wandb"
target_run_ids = ["0kQIoiOs", ]  # set run ids for e
metric4testing = "ent_offset_f1"
model_path_ids2infer = [0, 2, -1]
cal_scores = True  # set False if golden annotations are not give in data

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model settings >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
pos_tag_emb_config = {
    "pos_tag_num": statistics["pos_tag_num"],
    "emb_dim": 64,
    "emb_dropout": 0.1
} if pos_tag_emb else None

ner_tag_emb_config = {
    "ner_tag_num": statistics["ner_tag_num"],
    "emb_dim": 32,
    "emb_dropout": 0.1
} if ner_tag_emb else None

char_encoder_config = {
    "char_size": statistics["char_num"],
    "emb_dim": 16,
    "emb_dropout": 0.1,
    "bilstm_layers": [1, 1],  # layer num in bilstm1 and bilstm2
    "bilstm_hidden_size": [16, 32],  # hidden sizes of bilstm1 and bilstm2
    "bilstm_dropout": [0., 0.1, 0.],  # dropout rates for bilstm1, middle dropout layer, bilstm2
    "max_char_num_in_tok": 16,
} if char_encoder else None

word_encoder_config = {
    "word2id": dicts["word2id"],
    # eegcn_word_emb.txt
    "word_emb_file_path": "../../data/pretrained_emb/{}".format(pretrained_emb_name),
    "emb_dropout": 0.1,
    "bilstm_layers": [1, 1],
    "bilstm_hidden_size": [300, 600],
    "bilstm_dropout": [0., 0.1, 0.],
    "freeze_word_emb": False,
} if word_encoder else None

subwd_encoder_config = {
    "pretrained_model_path": pretrained_model_path,
    "finetune": True,
    "use_last_k_layers": 1,
} if subwd_encoder else None

dep_config = {
    "dep_type_num": statistics["deprel_type_num"],
    "dep_type_emb_dim": 50,
    "emb_dropout": 0.1,
    "gcn_dim": 128,
    "gcn_dropout": 0.1,
    "gcn_layer_num": 1,
} if dep_gcn else None

handshaking_kernel_config = {
    "shaking_type": "cln+bilstm"
}

# model settings
model_settings = {
    "pos_tag_emb_config": pos_tag_emb_config,
    "ner_tag_emb_config": ner_tag_emb_config,
    "char_encoder_config": char_encoder_config,
    "subwd_encoder_config": subwd_encoder_config,
    "word_encoder_config": word_encoder_config,
    "dep_config": dep_config,
    "handshaking_kernel_config": handshaking_kernel_config,
    "fin_hidden_size": 1024
}

model_settings_log = copy.deepcopy(model_settings)
if "word_encoder_config" in model_settings_log and model_settings_log["word_encoder_config"] is not None:
    del model_settings_log["word_encoder_config"]["word2id"]

# this dict would be logged
config_to_log = {
    "seed": seed,
    "task_type": task_type,
    "model_name": model_name,
    "tagger_name": tagger_name,
    "epochs": epochs,
    "learning_rate": lr,
    "batch_size_train": batch_size_train,
    "batch_size_valid": batch_size_valid,
    "batch_size_test": batch_size_test,
    "max_seq_len_train": max_seq_len_train,
    "max_seq_len_valid": max_seq_len_valid,
    "max_seq_len_test": max_seq_len_test,
    "sliding_len_train": sliding_len_train,
    "sliding_len_valid": sliding_len_valid,
    "sliding_len_test": sliding_len_test,
    "note": note,
    "model_state_dict_path": model_state_dict_path,
    **trainer_config,
    **model_settings_log,
    "token_level": token_level,
    "optimizer": optimizer_config,
    "addtional_preprocessing_config": addtional_preprocessing_config,
    "tagger_config": tagger_config,
    "combine_split": combine,
}
