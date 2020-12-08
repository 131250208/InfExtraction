import torch
import random
import numpy as np
import os

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

exp_name = "ace2005_lu"
task_type = "ee"

# model and tagger(decoder)
model_name = "TriggerFreeEventExtractor"
tagger_name = "MatrixTaggerEE"
# data
data_in_dir = "../../data/normal_data"
data_out_dir = "../../data/res_data"
train_data = os.path.join(data_in_dir, exp_name, "train_data.json")
valid_data = os.path.join(data_in_dir, exp_name, "valid_data.json")
test_data_list = glob("{}/*test*".format(os.path.join(data_in_dir, exp_name))) # ["test_triples.json", ], ["test_data.json", ]
dicts = "dicts.json"
statistics = "statistics.json"
statistics_path = os.path.join(data_in_dir, exp_name, statistics)
dicts_path = os.path.join(data_in_dir, exp_name, dicts)
statistics = json.load(open(statistics_path, "r", encoding="utf-8"))
dicts = json.load(open(dicts_path, "r", encoding="utf-8"))

# for preprocessing
key2dict = {
    "char_list": dicts["char2id"],
    "word_list": dicts["word2id"],
    "subword_list": dicts["bert_dict"],
    "pos_tag_list": dicts["pos_tag2id"],
    "ner_tag_list": dicts["ner_tag2id"],
    "dependency_list": dicts["deprel_type2id"],
}

# train, valid, test settings
run_name = "{}+{}+{}".format(task_type, re.sub("[^A-Z]", "", model_name), re.sub("[^A-Z]", "", tagger_name))
check_tagging_n_decoding = True
device_num = 1
epochs = 200
lr = 1e-4 # 5e-5, 1e-4
batch_size_train = 8
batch_size_valid = 8
batch_size_test = 8

max_seq_len_train = 64
max_seq_len_valid = 64
max_seq_len_test = 64

sliding_len_train = 64
sliding_len_valid = 64
sliding_len_test = 64

combine = True

scheduler = "CAWR"
use_ghm = False
final_score_key = "trigger_class_f1" # arg_class_f1
model_bag_size = 15
score_threshold = 0

# schedulers
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
use_wandb = True
log_interval = 10

default_run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
default_log_path = "./default_log_dir/default.log"
default_dir_to_save_model = "./default_log_dir/{}".format(default_run_id)

# training config
trainer_config = {
    "run_name": run_name,
    "exp_name": exp_name,
    "scheduler_config": scheduler_dict[scheduler],
    "log_interval": log_interval,
}

# pretrianed model state
# run-20201126_005415-29nsodmj/model_state_dict_0_18.75.pt
# ./wandb/run-20201126_003324-3c6z9kvu/model_state_dict_13_70.886.pt
model_state_dict_path = None 


# for test
model_dir_for_test = "wandb"  # "./default_log_dir"
target_run_ids = ["1zbzg5ml", "11p5ec06"]
top_k_models = 3
cal_scores = True # set False if the test sets are not annotated with golden results

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    "emb_dim": 16,
    "emb_dropout": 0.1,
    "bilstm_layers": [1, 1], # layer num in bilstm1 and bilstm2
    "bilstm_hidden_size": [16, 32], # hidden sizes of bilstm1 and bilstm2
    "bilstm_dropout": [0., 0.1, 0.], # dropout rates for bilstm1, middle dropout layer, bilstm2
    "max_char_num_in_tok": 16,
}

word_encoder_config = {
    "word2id": dicts["word2id"],
    # eegcn_word_emb.txt
    # 
    "word_emb_file_path": "../../data/pretrained_emb/glove.6B.100d.txt",
    "emb_dropout": 0.1,
    "bilstm_layers": [1, 1],
    "bilstm_hidden_size": [300, 600],
    "bilstm_dropout": [0., 0.1, 0.],
    "freeze_word_emb": False,
}

subwd_encoder_config = {
    "pretrained_model_path": "../../data/pretrained_models/bert-base-cased",
    "finetune": True,
    "use_last_k_layers": 1,
    "wordpieces_prefix": "##",
}

dep_config = {
    "dep_type_num": statistics["deprel_type_num"],
    "dep_type_emb_dim": 50,
    "emb_dropout": 0.1,
    "gcn_dim": 128,
    "gcn_dropout": 0.1,
    "gcn_layer_num": 1,
}

handshaking_kernel_config = {
    "shaking_type": "cat+cln",
}

conv_config = {
    "conv_layers": 3,
    "conv_kernel_size": 3,  # must be odd
}

token_level = "word" # token is word or subword
# subword: use bert tokenizer to get subwords, use stanza to get words, other features are aligned with the subwords
# word: use stanza to get words, wich can be fed into both bilstm and bert

# to do an ablation study, you can remove components by commenting the configurations below
# except for handshaking_kernel_config, which is a must for the model
model_settings = {
#     "pos_tag_emb_config": pos_tag_emb_config,
#     "ner_tag_emb_config": ner_tag_emb_config,
#     "char_encoder_config": char_encoder_config,
#     "subwd_encoder_config": subwd_encoder_config,
    "word_encoder_config": word_encoder_config,
#     "dep_config": dep_config,
    "handshaking_kernel_config": handshaking_kernel_config,
    "conv_config": conv_config,
    "fin_hidden_size": 512,
}
if model_name == "TPLinkerPP":
    assert max_seq_len_train == max_seq_len_valid == max_seq_len_test
    model_settings["matrix_size"] = max_seq_len_train

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> log >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
    "note": "",
    "model_state_dict_path": model_state_dict_path,
    **trainer_config,
    **model_settings_log,
    "token_level": token_level,
}

