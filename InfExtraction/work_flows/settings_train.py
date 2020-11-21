import string
import random
# data
data_in_dir = "../../data/normal_data"
train_data = "train_data.json"
valid_data = "train_data.json"
dicts = "dicts.json"
statistics = "statistics.json"

# training settings
exp_name = "ace2005_lu"
device_num = 0
task_type = "ee"
language = "en"
use_bert = True
seed = 2333
batch_size = 32
epochs = 200
log_interval = 10
max_seq_len = 100
sliding_len = 20
scheduler = "CAWR"
ghm = False
tok_pair_sample_rate = 1

# logger
run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
use_wandb = False
log_path = "./default_log_dir/default.log"
path_to_save_model = "./default_log_dir/{}".format(run_id)
note = ""

# model
run_name = ""
model_name = "tplinker_plus"

bilstm_settings = {
    "pretrained_word_embedding_path": "../../pretrained_emb/glove_300_nyt.emb",
    "lr": 1e-3,
    "lstm_l1_hidden_size": 300,
    "lstm_l2_hidden_size": 600,
    "emb_dropout": 0.1,
    "rnn_dropout": 0.1,
    "word_embedding_dim": 300,
}

pretrained_model_settings = {
    "wordpieces_prefix": "##",
    "lr": 5e-5,
    "pretrained_model_path": "../../data/pretrained_models/bert-base-uncased",
}

handshaking_kernal_settings = {
    "shaking_type": "cln_lstm",
}

model_settings = {
    **bilstm_settings,
    **handshaking_kernal_settings,
}
if use_bert:
    model_settings = {**model_settings, **pretrained_model_settings}

config_to_log = {
    "model_name": model_name,
    "seed": seed,
    "task_type": task_type,
    "batch_size": batch_size,
    "epochs": epochs,
    "log_interval": log_interval,
    "max_seq_len": max_seq_len,
    "sliding_len": sliding_len,
    "scheduler": scheduler,  # Step
    "ghm": ghm,
     "tok_pair_sample_rate": tok_pair_sample_rate,
    **model_settings,
}
# match_pattern:
# only_head_text (nyt_star, webnlg_star),
# whole_text (nyt, webnlg),
# only_head_index,
# whole_span,
# event_extraction
match_pattern = "event_extraction"