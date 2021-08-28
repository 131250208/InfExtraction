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
from InfExtraction.modules.utils import MyLargeFileReader, MyLargeJsonlinesFileReader, merge_gen

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
exp_name = "duee_comp2021"
load_data2memory = True
language = "ch"
task_type = "re+tfboys"  # re+tfboys4doc_ee, re+tfboys
model_name = "RAIN"
tagger_name = "Tagger4RAIN"
run_name = "{}+{}+{}".format(task_type, re.sub("[^A-Z]", "", model_name), re.sub("[^A-Z]", "", tagger_name))
pretrained_model_name = "macbert-large"
pretrained_emb_name = "glove_du_300.txt"
use_wandb = False
note = ""
epochs = 100
lr = 2e-5  # 5e-5, 1e-4
check_tagging_n_decoding = False
split_early_stop = True
drop_neg_samples = False
combine = False
scheduler = "CAWR"
use_ghm = False

metric_keyword = "arg_soft_class_char_level_f1"  # save models on which metric: f1, ...
model_bag_size = 20  # if no saving, set to 0

batch_size_train = 12
batch_size_valid = 4
batch_size_test = 4

max_seq_len_train = 72
max_seq_len_valid = 100
max_seq_len_test = 100

sliding_len_train = 20
sliding_len_valid = 50
sliding_len_test = 50

# >>>>>>>>>>>>>>>>> features >>>>>>>>>>>>>>>>>>>
token_level = "subword"  # token is word or subword

pos_tag_emb = True
ner_tag_emb = False
char_encoder = True
dep_gcn = True

word_encoder = True
subwd_encoder = True
use_attns4rel = True  # use only if subwd_encoder (bert) is True
flair = False
elmo = False

# >>>>>>>>>>>>>>>>>>>> data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
data_in_dir = "../../data/normal_data"
data_out_dir = "../../data/res_data"
train_path = os.path.join(data_in_dir, exp_name, "train_data.json")
val_path = os.path.join(data_in_dir, exp_name, "valid_data.json")
test_path_list = glob("{}/test_data_2.json".format(os.path.join(data_in_dir, exp_name)))

max_lines = None
train_lfreader = MyLargeFileReader(train_path, max_lines=max_lines)
train_jsreader = MyLargeJsonlinesFileReader(train_lfreader)
val_lfreader = MyLargeFileReader(val_path, max_lines=max_lines)
val_jsreader = MyLargeJsonlinesFileReader(val_lfreader)

train_data = merge_gen(train_jsreader.get_jsonlines_generator())
valid_data = val_jsreader.get_jsonlines_generator()
data4checking = val_jsreader.get_jsonlines_generator(end_idx=50)

filename2test_data = {}
for test_data_path in test_path_list:
    filename = test_data_path.split("/")[-1]
    test_lfreader = MyLargeFileReader(test_data_path, max_lines=max_lines)
    test_jsreader = MyLargeJsonlinesFileReader(test_lfreader)
    test_data = test_jsreader.get_jsonlines_generator()
    filename2test_data[filename] = test_data

dicts = "dicts.json"
statistics = "statistics.json"
statistics_path = os.path.join(data_in_dir, exp_name, statistics)
dicts_path = os.path.join(data_in_dir, exp_name, dicts)
statistics = json.load(open(statistics_path, "r", encoding="utf-8"))
dicts = json.load(open(dicts_path, "r", encoding="utf-8"))
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# for index_features()
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
    "add_default_entity_type": False,
    "classify_entities_by_relation": False,
    "dtm_arg_type_by_edges": False,
}

# tagger config
tagger_config = {
    "dtm_arg_type_by_edges": addtional_preprocessing_config["dtm_arg_type_by_edges"],
    "classify_entities_by_relation": addtional_preprocessing_config["classify_entities_by_relation"],
    "add_h2t_n_t2h_links": False,
    "language": language,
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
target_run_ids = ["mj9p5ngf", ]
model_path_ids2infer = range(8, 3, -1)
metric4testing = "arg_soft_class_char_level_f1"
cal_scores = True  # set False if the test sets are not annotated

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

pos_tag_emb_config = {
    "pos_tag_num": statistics["pos_tag_num"],
    "emb_dim": 64,
    "emb_dropout": 0.1,
    "hsk_emb": True,
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

flair_config = {
    "embedding_models": [
        {
            "model_name": "ELMoEmbeddings",
            "parameters": [],
        },
    ]
} if flair else None

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
    "gcn": False,
    "hnt_emb": True,
} if dep_gcn else None

handshaking_kernel_config = {
    "ent_shaking_type": "cln+bilstm",
    "rel_shaking_type": "cln",
    "ent_dist_emb_dim": 64,
    "rel_dist_emb_dim": 128,
}

# model settings
model_settings = {
    "pos_tag_emb_config": pos_tag_emb_config,
    "ner_tag_emb_config": ner_tag_emb_config,
    "char_encoder_config": char_encoder_config,
    "subwd_encoder_config": subwd_encoder_config,
    "word_encoder_config": word_encoder_config,
    "flair_config": flair_config,
    "dep_config": dep_config,
    "handshaking_kernel_config": handshaking_kernel_config,
    "use_attns4rel": use_attns4rel,
    "ent_dim": 1500,
    "rel_dim": 1500,
    "span_len_emb_dim": -1,
    "emb_ent_info2rel": False,
    "golden_ent_cla_guide": False,
    "loss_func": "mce_loss",
    "loss_weight": 0.5,
    "loss_weight_recover_steps": 0,
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
    "addtional_preprocessing_config": addtional_preprocessing_config,
    "tagger_config": tagger_config,
    "split_early_stop": split_early_stop,
    "drop_neg_samples": drop_neg_samples,
    "combine_split": combine,
    "use_ghm": use_ghm,
}
