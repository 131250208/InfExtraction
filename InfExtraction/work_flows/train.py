'''
Train the model

# load data
# Split the data
# Index the data
# Tag the data
# Init model
# Put the data into a Dataloaders
# Set optimizer and trainer
'''

from InfExtraction.modules.preprocess import Preprocessor, MyDataset
from InfExtraction.modules.taggers import HandshakingTagger4EE
from InfExtraction.modules.workers import Trainer
from InfExtraction.modules.models import TPLinkerPlus
from InfExtraction.work_flows import settings_train as settings
from InfExtraction.work_flows.utils import DefaultLogger

import os
import torch
import wandb
import json
from pprint import pprint
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # settings
    exp_name = settings.exp_name
    data_in_dir = settings.data_in_dir

    # data
    train_data_path = os.path.join(data_in_dir, exp_name, settings.train_data)
    valid_data_path = os.path.join(data_in_dir, exp_name, settings.valid_data)
    dicts_path = os.path.join(data_in_dir, exp_name, settings.dicts)
    statistics = settings.statistics

    use_wandb = settings.use_wandb
    run_name = settings.run_name
    config2log = settings.config_to_log
    # logger settings
    default_log_path = settings.default_log_path
    default_run_id = settings.default_run_id
    default_dir_to_save_model = settings.default_dir_to_save_model

    # training settings
    device_num = settings.device_num
    task_type = settings.task_type
    language = settings.language
    use_bert = settings.use_bert
    seed = settings.seed
    batch_size = settings.batch_size
    epochs = settings.epochs
    log_interval = settings.log_interval
    max_seq_len = settings.max_seq_len
    sliding_len = settings.sliding_len
    trainer_config = settings.trainer_config
    lr = settings.lr
    model_state_dict_path = settings.model_state_dict_path

    # model settings
    model_settings = settings.model_settings

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed) # pytorch random seed
    torch.backends.cudnn.deterministic = True # for reproductivity

    # reset settings from args
    # ...

    # load data
    train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
    valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
    dicts = json.load(open(dicts_path, "r", encoding="utf-8"))

    # logger
    if use_wandb:
        # init wandb
        wandb.init(project=exp_name, name=run_name, config=config2log)
        dir_to_save_model = wandb.run.dir
        logger = wandb
    else:
        logger = DefaultLogger(default_log_path,
                               exp_name,
                               run_name,
                               default_run_id,
                               config2log)
        dir_to_save_model = default_dir_to_save_model
        if not os.path.exists(dir_to_save_model):
            os.makedirs(dir_to_save_model)

    # splitting
    if use_bert:
        max_seq_len_statistics = statistics["max_subword_seq_length"]
        feature_list_key = "subword_level_features"
    else:
        max_seq_len_statistics = statistics["max_word_seq_length"]
        feature_list_key = "word_level_features"
    max_seq_len = min(max_seq_len, max_seq_len_statistics)
    split_train_data = Preprocessor.split_into_short_samples(train_data, max_seq_len, sliding_len, "train",
                                                             wordpieces_prefix=model_settings["subwd_encoder_config"]["wordpieces_prefix"],
                                                             feature_list_key=feature_list_key)
    split_valid_data = Preprocessor.split_into_short_samples(valid_data, max_seq_len_statistics, sliding_len, "valid",
                                                             wordpieces_prefix=model_settings["subwd_encoder_config"]["wordpieces_prefix"],
                                                             feature_list_key=feature_list_key)
    sample_id2dismatched = Preprocessor.check_splits(split_train_data)
    pprint(sample_id2dismatched)
    sample_id2dismatched = Preprocessor.check_splits(split_valid_data)
    pprint(sample_id2dismatched)

    # indexing
    key2dict = {
        "char_list": dicts["char2id"],
        "word_list": dicts["word2id"],
        "pos_tag_list": dicts["pos_tag2id"],
        "ner_tag_list": dicts["ner_tag2id"],
        "dependency_list": dicts["deprel_type2id"],
    }
    pretrained_model_padding = model_settings["pretrained_model_padding"] if "pretrained_model_padding" in model_settings else 0
    indexed_train_data = Preprocessor.index_features(split_train_data,
                                                     key2dict,
                                                     max_seq_len,
                                                     model_settings["char_encoder_config"]["max_char_num_in_tok"],
                                                     pretrained_model_padding)
    indexed_valid_data = Preprocessor.index_features(split_valid_data,
                                                     key2dict,
                                                     max_seq_len_statistics,
                                                     model_settings["char_encoder_config"]["max_char_num_in_tok"],
                                                     pretrained_model_padding)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # tagging
    tagger = HandshakingTagger4EE(dicts["rel_type2id"], dicts["ent_type2id"])
    indexed_train_data = tagger.tag(indexed_train_data)
    indexed_valid_data = tagger.tag(indexed_valid_data)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # model
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    model_settings["word_encoder_config"]["word2id"] = dicts["word2id"] # set word2id dict
    tag_size = tagger.get_tag_size()
    model = TPLinkerPlus(tag_size, tagger, **model_settings)
    collate_fn = model.generate_batch
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # dataset
    train_dataset = MyDataset(indexed_train_data)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=False,
                                  collate_fn=collate_fn,
                                 )
    valid_dataset = MyDataset(indexed_valid_data)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=False,
                                  collate_fn=collate_fn,
                                 )

    # # # have a look at dataloader
    # train_data_iter = iter(train_dataloader)
    # batch_data = next(train_data_iter)
    # print(batch_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    if model_state_dict_path is not None:
        model.load_state_dict(torch.load(model_state_dict_path))
        print("------------model state {} loaded ----------------".format(model_state_dict_path.split("/")[-1]))

    # trainer
    trainer = Trainer(model, device, optimizer, trainer_config, logger, dir_to_save_model)
    trainer.train_n_valid(train_dataloader, valid_dataloader, epochs)