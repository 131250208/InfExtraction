import string
import random
import os
import json
import copy

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
dicts_path = os.path.join(data_in_dir, exp_name, dicts)
statistics = json.load(open(statistics_path, "r", encoding="utf-8"))
dicts = json.load(open(dicts_path, "r", encoding="utf-8"))

# for preprocessing
token_level = "subword"
key2dict = {
    "char_list": dicts["char2id"],
    "word_list": dicts["word2id"],
    "subword_list": dicts["bert_dict"],
    "pos_tag_list": dicts["pos_tag2id"],
    "ner_tag_list": dicts["ner_tag2id"],
    "dependency_list": dicts["deprel_type2id"],
}

# train, valid, test settings
run_name = "tp2+dep+pos+ner"
model_name = "tplinker_plus"
device_num = 1
# use_bert = True
seed = 2333
epochs = 200
lr = 5e-5
batch_size_train = 8
batch_size_valid = 32
batch_size_test = 32

max_seq_len_train = 64
max_seq_len_valid = 64
max_seq_len_test = 64

sliding_len_train = 20
sliding_len_valid = 20
sliding_len_test = 20

scheduler = "CAWR"
use_ghm = True
score_threshold = 0

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

# logger
use_wandb = False
log_interval = 10

default_run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
default_log_path = "./default_log_dir/default.log"
default_dir_to_save_model = "./default_log_dir/{}".format(default_run_id)

# training config
trainer_config = {
    "task_type": task_type,
    "run_name": run_name,
    "exp_name": exp_name,
    "scheduler_config": scheduler_dict[scheduler],
    "use_ghm": use_ghm,
    "log_interval": log_interval,
}

# for eval
score_threshold = score_threshold
model_bag_size = 15

# pretrianed model state
# dep b8: run-20201123_122929-1zbzg5ml/model_state_dict_11.pt
# dep b32: run-20201123_123852-11p5ec06/model_state_dict_18.pt

model_state_dict_path = None # "./wandb/run-20201123_122929-1zbzg5ml/model_state_dict_11.pt"


# for test
model_dir_for_test = "./wandb" # "./default_log_dir"
target_run_ids = ["1zbzg5ml", "11p5ec06"]
top_k_models = 3
cal_scores = True # set False if the test sets are not annotated with golden results

# model
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
    "word2id": dicts["word2id"],
    # eegcn_word_emb.txt
    # PubMed-shuffle-win-30.bin
    "word_emb_file_path": "../../data/pretrained_emb/eegcn_word_emb.txt",
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
    "wordpieces_prefix": "##",
}

dep_config = {
    "dep_type_num": statistics["deprel_type_num"],
    "dep_type_emb_dim": 64,
    "emb_dropout": 0.1,
    "gcn_dim": 256,
    "gcn_dropout": 0.1,
    "gcn_layer_num": 2,
}

handshaking_kernel_config = {
    "shaking_type": "cln",
}

# set None to rm a component
model_settings = {
    "pos_tag_emb_config": pos_tag_emb_config,
    "ner_tag_emb_config": ner_tag_emb_config, 
    "char_encoder_config": char_encoder_config,
    "subwd_encoder_config": subwd_encoder_config,
    "word_encoder_config": word_encoder_config,
    "dep_config": dep_config,
    "handshaking_kernel_config": handshaking_kernel_config,
    "fin_hidden_size": 1024,
}

# this dict would be logged
model_settings_log = copy.deepcopy(model_settings)
if model_settings_log["word_encoder_config"] is not None:
    del model_settings_log["word_encoder_config"]["word2id"]
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
    "model_state_dict_path": model_state_dict_path,
    **model_settings_log,
}
# match_pattern: for joint entity and relation extraction
# only_head_text (nyt_star, webnlg_star),
# whole_text (nyt, webnlg),
# only_head_index,
# whole_span
match_pattern = None
