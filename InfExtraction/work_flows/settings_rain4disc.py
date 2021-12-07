import os
device_num = 1
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
import torch
import random
import numpy as np
from datetime import date
import time
from InfExtraction.modules.utils import load_data
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
exp_name = "cadec4yelp"
language = "en"
task_type = "re+ner"  # re, re+ee
model_name = "RAIN"
tagger_name = "Tagger4RAIN"
run_name = "{}+{}+{}".format(task_type, re.sub("[^A-Z]", "", model_name), re.sub("[^A-Z]", "", tagger_name))
pretrained_model_name = "yelpbert"
pretrained_emb_name = "glove.6B.100d.txt"
use_wandb = True
note = ""
epochs = 300
lr = 1e-5  # 5e-5, 1e-4
check_tagging_n_decoding = False
combine = True  # combine splits
scheduler = "CAWR"
model_bag_size = 0
metric_pattern2save = None  # if none save best models on all metrics

batch_size_train = 12
batch_size_valid = 12
batch_size_test = 12

max_seq_len_train = 64
max_seq_len_valid = 100
max_seq_len_test = 100

sliding_len_train = 64
sliding_len_valid = 100
sliding_len_test = 100

# >>>>>>>>>>>>>>>>> features >>>>>>>>>>>>>>>>>>>
token_level = "subword"  # token is word or subword

# to do an ablation study, you can ablate components by setting it to False
pos_tag_emb = False
ner_tag_emb = False
char_encoder = False
dep_gcn = False

word_encoder = False
subwd_encoder = True
use_attns4rel = True

# data
data_in_dir = "../../data/preprocessed_data"
data_out_dir = "../../data/res_data"

train_data_path = os.path.join(data_in_dir, exp_name, "train_data.json")
valid_data_path = os.path.join(data_in_dir, exp_name, "valid_data.json")
test_data_path_list = glob("{}/*test*.json".format(os.path.join(data_in_dir, exp_name)))

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
    # "bert_dict": "subword_list",
    "pos_tag2id": "pos_tag_list",
    "ner_tag2id": "ner_tag_list",
    "deprel_type2id": "dependency_list",
}
key2dict = {
    "subword_list": BertTokenizer.from_pretrained("../../data/pretrained_models/{}".format(pretrained_model_name)).get_vocab()
}
for key, val in dicts.items():
    if key in key_map:
        key2dict[key_map[key]] = val

# additional preprocessing
addtional_preprocessing_config = {
    "add_default_entity_type": False,
    "classify_entities_by_relation": False,
    "use_bound": True,
}

# tagger config
tagger_config = {
    "classify_entities_by_relation": addtional_preprocessing_config["classify_entities_by_relation"],
    "add_h2t_n_t2h_links": True,
    "add_o2s_links": False,
    "language": language,
    "use_bound": addtional_preprocessing_config["use_bound"],
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
    "pretrained_model_path": "../../data/pretrained_models/{}".format(pretrained_model_name),
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
    "tok_pair_neg_sampling_rate": .5,
    "clique_comp_loss": False,
    "do_span_len_emb": True,
    "loss_weight": 0.5,
    "loss_weight_recover_steps": 0,
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
    "combine_split": combine,
}
