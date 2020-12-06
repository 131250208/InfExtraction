import string
import random
import os
import json
import copy
import re

exp_name = "nyt"
task_type = "re"
# match_pattern: for joint entity and relation extraction
# only_head_text (nyt_star, webnlg_star),
# whole_text (nyt, webnlg),
# only_head_index,
# whole_span
match_pattern = "whole_text"

# model and tagger(decoder)
model_name = "TPLinkerPP" # TPLinkerPlus, TPLinkerPP, TriggerFreeEventExtractor
tagger_name = "HandshakingTaggerRel4TPLPP" # HandshakingTaggerRel4TPLPlus, HandshakingTaggerRel4TPLPP, HandshakingTaggerEE4TPLPlus, MatrixTaggerEE

# data
data_in_dir = "../../data/normal_data"
data_out_dir = "../../data/res_data"
train_data = "train_data.json"
valid_data = "valid_data.json"
test_data_list = ["test.json", ] # ["test_triples.json", ], ["test_data.json", ]
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
device_num = 0
seed = 2333
epochs = 200
lr = 5e-5 # 5e-5, 1e-4
batch_size_train = 24
batch_size_valid = 32
batch_size_test = 32

max_seq_len_train = 100
max_seq_len_valid = 100
max_seq_len_test = 100

sliding_len_train = 20
sliding_len_valid = 20
sliding_len_test = 20

combine = False

scheduler = "CAWR"
use_ghm = False
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
use_wandb = False
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

# for eval
final_score_key = "rel_f1" # trigger_class_f1
score_threshold = score_threshold
model_bag_size = 15

# pretrianed model state
# run-20201126_005415-29nsodmj/model_state_dict_0_18.75.pt
# ./wandb/run-20201126_003324-3c6z9kvu/model_state_dict_13_70.886.pt
model_state_dict_path = None 


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
#     "shaking_type": "cln",
    "ent_shaking_type": "cln_lstm",
    "rel_shaking_type": "cln",
}

# model settings
token_level = "subword" # token is word or subword
# subword: use bert tokenizer to get subwords, use stanza to get words, other features are aligned with the subwords
# word: use stanza to get words, wich can be fed into both bilstm and bert
# to do an ablation study, you can remove components by commenting the configurations below
# except for handshaking_kernel_config, which is a must for the model
model_settings = {
#     "pos_tag_emb_config": pos_tag_emb_config,
#     "ner_tag_emb_config": ner_tag_emb_config,
#     "char_encoder_config": char_encoder_config,
    "subwd_encoder_config": subwd_encoder_config,
#     "word_encoder_config": word_encoder_config,
#     "dep_config": dep_config,
    "handshaking_kernel_config": handshaking_kernel_config,
#     "fin_hidden_size": 1024,
    "ent_fc_in_dim": 512,
    "rel_fc_in_dim": 768,
}

# this dict would be logged
model_settings_log = copy.deepcopy(model_settings)
if "word_encoder_config" in model_settings_log and model_settings_log["word_encoder_config"] is not None:
    del model_settings_log["word_encoder_config"]["word2id"]

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

