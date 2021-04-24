import os
device_num = 0
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
import torch
import random
import numpy as np
from datetime import date
import time
from InfExtraction.modules.utils import load_data

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
stage = "train"  # inference
task_type = "ner"
model_name = "SpanNER"
tagger_name = "Tagger4SpanNER"
run_name = "{}+{}+{}".format(task_type, re.sub("[^A-Z]", "", model_name), re.sub("[^A-Z]", "", tagger_name))
pretrained_model_name = "biobert-large-cased-pubmed-58k"
pretrained_emb_name = "glove.6B.100d.txt"
use_wandb = True
note = ""
epochs = 100
lr = 3e-5  # 5e-5, 1e-4
check_tagging_n_decoding = True
split_early_stop = True
drop_neg_samples = False
combine = True  # combine splits
scheduler = "CAWR"
use_ghm = False
model_bag_size = 0

batch_size_train = 8
batch_size_valid = 12
batch_size_test = 12 

max_seq_len_train = 64
max_seq_len_valid = 64
max_seq_len_test = 64

sliding_len_train = 20
sliding_len_valid = 20
sliding_len_test = 20

# >>>>>>>>>>>>>>>>> features >>>>>>>>>>>>>>>>>>>
token_level = "subword"  # token is word or subword

# to do an ablation study, you can ablate components by setting it to False
pos_tag_emb = False
ner_tag_emb = False
char_encoder = False
dep_gcn = False

word_encoder = False
subwd_encoder = True
flair = False
elmo = False
top_attn = False

# data
data_in_dir = "../../data/normal_data"
data_out_dir = "../../data/res_data"
train_path = os.path.join(data_in_dir, exp_name, "train_data.json")
val_path = os.path.join(data_in_dir, exp_name, "valid_data.json")
max_lines = None  # None
train_data = load_data(train_path, lines=max_lines)
valid_data = load_data(val_path, lines=max_lines)

checking_num = 1000
data4checking = copy.deepcopy(valid_data[:checking_num])
random.shuffle(data4checking)

test_data_list = [] # glob("{}/*test*.json".format(os.path.join(data_in_dir, exp_name)))
filename2ori_test_data = {}
for test_data_path in test_data_list:
    filename = test_data_path.split("/")[-1]
    ori_test_data = load_data(test_data_path, lines=max_lines)
    filename2ori_test_data[filename] = ori_test_data

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

# pretrianed model state
model_state_dict_path = None

# for test
model_dir_for_test = "./wandb"  # "./default_log_dir", "./wandb"
target_run_ids = ["eyu8cm6x", ]
top_k_models = 1
metric4testing = "ent_exact_offset_f1"
main_test_set_name = "test_data.json"
cal_scores = True  # set False if the test sets are not annotated

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

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
    "word_fusion_dim": 150,
    "word_emb_file_path": "../../data/pretrained_emb/{}".format(pretrained_emb_name),
    "emb_dropout": 0.1,
    "bilstm_layers": [1, 1],
    "bilstm_hidden_size": [300, 600],
    "bilstm_dropout": [0., 0.1, 0.],
    "freeze_word_emb": False,
} if word_encoder else None

flair_config = {
    "embedding_models": [
        {
            "model_name": "ELMoEmbeddings",
            "parameters": ["5.5B"],
        },
    ]
} if flair else None

elmo_config = {
    "model": "5.5B",
    "finetune": False,
    "dropout": 0.1,
} if elmo else None

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
    "shaking_type": "cln+bilstm"
}

# model settings
model_settings = {
    "pos_tag_emb_config": pos_tag_emb_config,
    "ner_tag_emb_config": ner_tag_emb_config,
    "char_encoder_config": char_encoder_config,
    "subwd_encoder_config": subwd_encoder_config,
    "word_encoder_config": word_encoder_config,
    "flair_config": flair_config,
    "elmo_config": elmo_config,
    "dep_config": dep_config,
    "handshaking_kernel_config": handshaking_kernel_config,
    "ent_dim": 1024,
    "span_len_emb_dim": 64,
    "loss_func": "mce_loss",
    "pred_threshold": 0.,
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
    # "addtional_preprocessing_config": addtional_preprocessing_config,
    # "tagger_config": tagger_config,
    "split_early_stop": split_early_stop,
    "drop_neg_samples": drop_neg_samples,
    "combine_split": combine,
    "use_ghm": use_ghm,
}
