import string
import random
import os
import json

exp_name = "ace2005_lu"
task_type = "ee"

# data
data_in_dir = "../../data/normal_data"
data_out_dir = "../../data/res_data"
train_data = "train_data.json"
valid_data = "valid_data.json"
test_data_list = ["test_data.json", ]
dicts = "dicts.json"
statistics = "statistics.json"
statistics_path = os.path.join(data_in_dir, exp_name, statistics)
statistics = json.load(open(statistics_path, "r", encoding="utf-8"))

# train, valid, test settings
device_num = 1
use_bert = True
seed = 2333
epochs = 200
batch_size_train = 8
batch_size_valid = 32
batch_size_test = 32

max_seq_len_train = 100
max_seq_len_valid = 100
max_seq_len_test = 512

sliding_len_train = 20
sliding_len_valid = 20
sliding_len_test = 20

lr = 5e-5

scheduler = "CAWR"
use_ghm = False
score_threshold = 0
model_state_dict_path = None

# logger
use_wandb = True
log_interval = 10

default_run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
default_log_path = "./default_log_dir/default.log"
default_dir_to_save_model = "./default_log_dir/{}".format(default_run_id)

# model
run_name = "tp2+dep+pos+ner"
model_name = "tplinker_plus"

pos_tag_emb_config = {
    "pos_tag_num": statistics["pos_tag_num"],
    "emb_dim": 64,
    "emb_dropout": 0.1
}

ner_tag_emb_config = {
    "ner_tag_num": statistics["ner_tag_num"],
    "emb_dim": 32,
    "emb_dropout": 0.1
}

char_encoder_config = {
    "char_size": statistics["char_num"],
    "emb_dim": 32,
    "emb_dropout": 0.1,
    "bilstm_layers": [1, 1], # layer num in bilstm1 and bilstm2
    "bilstm_hidden_size": [32, 64], # hidden sizes of bilstm1 and bilstm2
    "bilstm_dropout": [0., 0.1, 0.], # dropout rates for bilstm1, middle dropout layer, bilstm2
    "max_char_num_in_tok": 16,
}

word_encoder_config = {
    "word_emb_file_path": "../../data/pretrained_emb/glove.6B.100d.txt", # '../../data/pretrained_emb/PubMed-shuffle-win-30.bin'
    "emb_dropout": 0.1,
    "bilstm_layers": [1, 1],
    "bilstm_hidden_size": [300, 600],
    "bilstm_dropout": [0., 0.1, 0.],
    "freeze_word_emb": False,
}

subwd_encoder_config = {
    "pretrained_model_path": "../../data/pretrained_models/bert-base-uncased",
    "finetune": True,
    "use_last_k_layers": 1,
    "pretrained_model_padding": 0,
    "wordpieces_prefix": "##",
}

dep_config = {
    "dep_type_num": statistics["deprel_type_num"],
    "dep_type_emb_dim": 64,
    "emb_dropout": 0.1,
    "gcn_dim": 128,
    "gcn_dropout": 0.1,
    "gcn_layer_num": 2,
}

handshaking_kernel_config = {
    "shaking_type": "cln",
}

model_settings = {
    "pos_tag_emb_config": pos_tag_emb_config,
    "ner_tag_emb_config": ner_tag_emb_config,
    "char_encoder_config": char_encoder_config,
    "word_encoder_config": word_encoder_config,
    "dep_config": dep_config,
    "handshaking_kernel_config": handshaking_kernel_config,
    "fin_hidden_size": 768,
}

if use_bert:
    model_settings["subwd_encoder_config"] = subwd_encoder_config

# schedulers
scheduler_dict = {
    "CAWR": {
        # CosineAnnealingWarmRestarts
        "name": "CAWR",
        "T_mult": 1,
        "rewarm_steps": 4000,
    },
    "StepLR": {
        "name": "StepLR",
        "decay_rate": 0.999,
        "decay_steps": 100,
    },
}

# training config
trainer_config = {
    "run_name": run_name,
    "exp_name": exp_name,
    "score_threshold": score_threshold,
    "scheduler_config": scheduler_dict[scheduler],
    "use_ghm": use_ghm,
    "log_interval": log_interval,
}

# this dict would be logged
config_to_log = {
    "model_name": model_name,
    "seed": seed,
    "task_type": task_type,
    "epochs": epochs,
    "batch_size_train": batch_size_train,
    "batch_size_valid": batch_size_valid,
    "max_seq_len_train": max_seq_len_train,
    "sliding_len_train": sliding_len_train,
    "max_seq_len_valid": max_seq_len_valid,
    "sliding_len_valid": sliding_len_valid,
    "note": "",
    **model_settings,
}
# match_pattern: for joint entity and relation extraction
# only_head_text (nyt_star, webnlg_star),
# whole_text (nyt, webnlg),
# only_head_index,
# whole_span
match_pattern = None
