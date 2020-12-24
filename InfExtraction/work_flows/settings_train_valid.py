import torch
import random
import numpy as np
import os
from datetime import date
import time

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

# if task_type == "re":
#     final_score_key = "rel_f1"
# elif task_type == "re+ee":
#     final_score_key = "trigger_class_f1"
# elif task_type == "re+ner":
#     final_score_key = "ent_f1"

# match_pattern: for joint entity and relation extraction
# only_head_text (nyt_star, webnlg_star),
# whole_text (nyt, webnlg),
# only_head_index,
# whole_span
# match_pattern = "whole_span"

# Frequent changes
exp_name = "webnlg"
stage = "train"  # inference
task_type = "re"  # re, re+ee
model_name = "TPLinkerPP"
tagger_name = "HandshakingTagger4TPLPP"
pretrained_model_name = "bert-base-cased"
pretrained_emb_name = "glove.6B.100d.txt"
device_num = 1
use_wandb = False
note = ""
epochs = 100
lr = 5e-5  # 5e-5, 1e-4
batch_size_train = 6
batch_size_valid = 6
batch_size_test = 6

max_seq_len_train = 100
max_seq_len_valid = 100
max_seq_len_test = 100

sliding_len_train = 20
sliding_len_valid = 20
sliding_len_test = 20

# data
data_in_dir = "../../data/normal_data"
data_out_dir = "../../data/res_data"
train_data = os.path.join(data_in_dir, exp_name, "train_data.json")
valid_data = os.path.join(data_in_dir, exp_name, "valid_data.json")
test_data_list = glob("{}/*test*.json".format(os.path.join(data_in_dir, exp_name))) # ["test_triples.json", ], ["test_data.json", ]
dicts = "dicts.json"
statistics = "statistics.json"
statistics_path = os.path.join(data_in_dir, exp_name, statistics)
dicts_path = os.path.join(data_in_dir, exp_name, dicts)
statistics = json.load(open(statistics_path, "r", encoding="utf-8"))
dicts = json.load(open(dicts_path, "r", encoding="utf-8"))

# for preprocessing
key_map = {
    "char2id": "char_list",
    "word2id": "word_list",
    "bert_dict": "subword_list",
    "pos_tag2id": "pos_tag_list",
    "ner_tag2id": "ner_tag_list",
    "deprel_type2id": "dependency_list",
}
key2dict = {}
for key, val in dicts.items():
    key2dict[key_map[key]] = val

# additional preprocessing
addtional_preprocessing_config = {
    "add_default_entity_type": True,
    "classify_entities_by_relation": False,  # ee, re
    "add_nested_relation": False,  # ner
    "add_same_type_relation": False,  # ner
}

# tagger config
tagger_config = {
    "classify_entities_by_relation": addtional_preprocessing_config["classify_entities_by_relation"],
    "link_all_related_tokens": False,
}

# train, valid, test settings
run_name = "{}+{}+{}".format(task_type, re.sub("[^A-Z]", "", model_name), re.sub("[^A-Z]", "", tagger_name))
check_tagging_n_decoding = True

split_early_stop = True
combine = False

scheduler = "CAWR"
use_ghm = False

model_bag_size = 25

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
        "rewarm_epochs": 2,
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
default_run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
default_log_path = "./default_log_dir/default.log"
default_dir_to_save_model = "./default_log_dir/run-{}_{}-{}".format(date.today().strftime("%Y%m%d"),
                                                                    str(time.time())[:6],
                                                                    default_run_id)
# trainer config
trainer_config = {
    "run_name": run_name,
    "exp_name": exp_name,
    "scheduler_config": scheduler_dict[scheduler],
    "log_interval": log_interval,
}

# pretrianed model state
model_state_dict_path = None 

# for test
model_dir_for_test = "./default_log_dir"  # "./default_log_dir", "./wandb"
target_run_ids = ["0kQIoiOs", ]
top_k_models = 1
metric4testing = "trigger_class_f1"
cal_scores = True  # set False if the test sets are not annotated

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# to do an ablation study, you can ablate components by setting it to False

pos_tag_emb = False
ner_tag_emb = False
char_encoder = False

word_encoder = False
subwd_encoder = True

dep_gcn = False
use_attns4rel = False

token_level = "subword" # token is word or subword
# subword: use bert tokenizer to get subwords, use stanza to get words, other features are aligned with the subwords
# word: use stanza to get words, wich can be fed into both bilstm and bert

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
    "bilstm_layers": [1, 1], # layer num in bilstm1 and bilstm2
    "bilstm_hidden_size": [16, 32], # hidden sizes of bilstm1 and bilstm2
    "bilstm_dropout": [0., 0.1, 0.], # dropout rates for bilstm1, middle dropout layer, bilstm2
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
    "pretrained_model_path": "../../data/pretrained_models/{}".format(pretrained_model_name),
    "finetune": True,
    "use_last_k_layers": 1,
    "wordpieces_prefix": "##",
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
    "ent_shaking_type": "cln+lstm",
    "rel_shaking_type": "cln",
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
    "use_attns4rel": use_attns4rel,
    "ent_dim": 768,
    "rel_dim": 768,
    "emb_ent_info2rel": True,
    "golden_ent_cla_guide": True,
    "loss_weight_recover_steps": 6000,
}

model_settings_log = copy.deepcopy(model_settings)
if "word_encoder_config" in model_settings_log and model_settings_log["word_encoder_config"] is not None:
    del model_settings_log["word_encoder_config"]["word2id"]

# this dict would be logged
config_to_log = {
    "model_name": model_name,
    "seed": seed,
    "task_type": task_type,
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
    "split_early_stop": split_early_stop,
    "combine_split": combine,
    "use_ghm": use_ghm,
}

