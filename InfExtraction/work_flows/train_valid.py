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
import sys, getopt
import importlib

# settings
try:
    opts, args = getopt.getopt(sys.argv[1:], "s:", ["settings="])
except getopt.GetoptError:
    print('test.py -s <settings_file> --settings <settings_file>')
    sys.exit(2)

settings_name = "settings_default"
for opt, arg in opts:
    if opt in ("-s", "--settings"):
        settings_name = arg

module_name = "InfExtraction.work_flows.{}".format(settings_name)
settings = importlib.import_module(module_name)

import os
import wandb
from pprint import pprint
import logging
import re
from glob import glob
import numpy as np

import copy
import torch
from torch.utils.data import DataLoader
from InfExtraction.modules.preprocess import Preprocessor
from InfExtraction.modules import taggers
from InfExtraction.modules import models
from InfExtraction.modules.workers import Trainer, Evaluator
from InfExtraction.modules.metrics import MetricsCalculator
from InfExtraction.modules.utils import DefaultLogger, MyDataset, load_data, save_as_json_lines
import random
import json


def get_dataloader(data,
                   language,
                   data_type,
                   token_level,
                   max_seq_len,
                   sliding_len,
                   combine,
                   batch_size,
                   key2dict,
                   tagger,
                   collate_fn,
                   max_char_num_in_tok=None,
                   ):
    # split test data
    data = Preprocessor.split_into_short_samples(data,
                                                 max_seq_len,
                                                 sliding_len,
                                                 data_type,
                                                 token_level)

    if combine:
        data = Preprocessor.combine(data, max_seq_len)

    # check spans
    sample_id2mismatched = Preprocessor.check_spans(data, language)
    if len(sample_id2mismatched) > 0:
        logging.warning("mismatch errors in {}".format(data_type))
        pprint(sample_id2mismatched)

    # inexing
    indexed_data = Preprocessor.index_features(data,
                                               language,
                                               key2dict,
                                               max_seq_len,
                                               max_char_num_in_tok)
    # tagging
    indexed_data = tagger.tag(indexed_data)

    # dataloader
    dataloader = DataLoader(MyDataset(indexed_data),
                            batch_size=batch_size,
                            shuffle=True,
                            # num_workers=0,
                            drop_last=False,
                            collate_fn=collate_fn,
                            worker_init_fn=worker_init_fn
                            )
    return dataloader


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_score_fr_path(model_path):
    return float(re.search("_([\d\.]+)\.pt", model_path.split("/")[-1]).group(1))


def get_last_k_paths(path_list, k):
    path_list = sorted(path_list, key=get_score_fr_path)
    return path_list[-k:]


if __name__ == "__main__":
    # task
    exp_name = settings.exp_name
    task_type = settings.task_type
    run_name = settings.run_name
    model_name = settings.model_name
    tagger_name = settings.tagger_name
    language = settings.language

    # data
    data_in_dir = settings.data_in_dir
    data_out_dir = settings.data_out_dir
    train_data = load_data(settings.train_data_path)
    valid_data = load_data(settings.valid_data_path)
    checking_num = 1000
    data4cheking = copy.deepcopy(train_data)
    random.shuffle(data4cheking)
    data4checking = data4cheking[:checking_num]
    filename2test_data = {}
    for test_data_path in settings.test_data_path_list:
        filename = test_data_path.split("/")[-1]
        test_data = load_data(test_data_path)
        filename2test_data[filename] = test_data

    dicts = settings.dicts
    statistics = settings.statistics
    key2dict = settings.key2dict  # map from feature key to indexing dict

    # additonal preprocessing config
    try:
        addtional_preprocessing_config = settings.addtional_preprocessing_config
    except Exception as e:
        addtional_preprocessing_config = {}

    # tagger config
    try:
        tagger_config = settings.tagger_config
    except:
        tagger_config = {}

    # logger settings
    use_wandb = settings.use_wandb
    config2log = settings.config_to_log
    default_log_path = settings.default_log_path
    default_run_id = settings.default_run_id
    default_dir_to_save_model = settings.default_dir_to_save_model
    log_interval = settings.log_interval

    # training settings
    check_tagging_n_decoding = settings.check_tagging_n_decoding
    device_num = settings.device_num
    token_level = settings.token_level
    epochs = settings.epochs
    batch_size_train = settings.batch_size_train
    max_seq_len_train = settings.max_seq_len_train
    sliding_len_train = settings.sliding_len_train

    batch_size_valid = settings.batch_size_valid
    max_seq_len_valid = settings.max_seq_len_valid
    sliding_len_valid = settings.sliding_len_valid

    batch_size_test = settings.batch_size_test
    max_seq_len_test = settings.max_seq_len_test
    sliding_len_test = settings.sliding_len_test

    combine = settings.combine

    trainer_config = settings.trainer_config
    lr = settings.lr
    model_state_dict_path = settings.model_state_dict_path  # pretrained model state

    # optimizer
    optimizer_config = settings.optimizer_config

    # test settings
    model_dir_for_test = settings.model_dir_for_test
    target_run_ids = settings.target_run_ids
    model_path_ids2infer = settings.model_path_ids2infer
    metric_pattern2save = settings.metric_pattern2save
    metric4testing = settings.metric4testing
    cal_scores = settings.cal_scores

    # save model
    model_bag_size = settings.model_bag_size

    # model settings
    model_settings = settings.model_settings
    
    max_char_num_in_tok = None
    if "char_encoder_config" in model_settings and model_settings["char_encoder_config"] is not None:
        max_char_num_in_tok = model_settings["char_encoder_config"]["max_char_num_in_tok"]

    do_lower_case = False
    if "subwd_encoder_config" in model_settings and \
            model_settings["subwd_encoder_config"] is not None and \
            "do_lower_case" in model_settings["subwd_encoder_config"]:
        do_lower_case = model_settings["subwd_encoder_config"]["do_lower_case"]

    # env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # reset settings from args
    # to do

    # choose features and spans by token level
    train_data = Preprocessor.choose_features_by_token_level(train_data, token_level, do_lower_case)
    train_data = Preprocessor.choose_spans_by_token_level(train_data, token_level)

    valid_data = Preprocessor.choose_features_by_token_level(valid_data, token_level, do_lower_case)
    valid_data = Preprocessor.choose_spans_by_token_level(valid_data, token_level)

    data4checking = Preprocessor.choose_features_by_token_level(data4checking, token_level, do_lower_case)
    data4checking = Preprocessor.choose_spans_by_token_level(data4checking, token_level)
    for filename, test_data in filename2test_data.items():
        filename2test_data[filename] = Preprocessor.choose_features_by_token_level(test_data, token_level, do_lower_case)
        filename2test_data[filename] = Preprocessor.choose_spans_by_token_level(test_data, token_level)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # tagger
    tagger_class_name = getattr(taggers, tagger_name)
    if task_type == "re+ee":
        tagger_class_name = taggers.create_rebased_ee_tagger(tagger_class_name)
    elif task_type == "re+oie":
        tagger_class_name = taggers.create_rebased_oie_tagger(tagger_class_name)
    elif task_type == "re+ner":
        tagger_class_name = taggers.create_rebased_discontinuous_ner_tagger(tagger_class_name)
    elif task_type == "re+tfboys":
        tagger_class_name = taggers.create_rebased_tfboys_tagger(tagger_class_name)

    # additional preprocessing
    def additional_preprocess(data):
        return tagger_class_name.additional_preprocess(data, **addtional_preprocessing_config)

    train_data = additional_preprocess(train_data)

    all_data4gen_tag_dict = []
    all_data4gen_tag_dict.extend(train_data)

    # tagger
    tagger = tagger_class_name(all_data4gen_tag_dict, **tagger_config)

    # model
    print("init model...")
    model_class_name = getattr(models, model_name)
    model = model_class_name(tagger, **model_settings)
    model = model.to(device)
    print("done!")

    # function for generating data batch
    collate_fn = model.generate_batch
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # logger
    if use_wandb:
        # init wandb
        wandb.init(project=exp_name, name=run_name, config=config2log,
                   # settings=wandb.Settings(start_method="fork"),
                   )
        dir_to_save_model = wandb.run.dir
        logger = wandb
        run_id = wandb.run.id
    else:
        logger = DefaultLogger(default_log_path,
                               exp_name,
                               run_name,
                               default_run_id,
                               config2log)
        run_id = default_run_id
        dir_to_save_model = default_dir_to_save_model
        if not os.path.exists(dir_to_save_model):
            os.makedirs(dir_to_save_model)

    # trainer and evaluator
    evaluator = Evaluator(model, device)

    # test dataloader
    filename2test_data_loader = {}
    for filename, test_data in filename2test_data.items():
        test_dataloader = get_dataloader(test_data,
                                         language,
                                         "test",
                                         token_level,
                                         max_seq_len_test,
                                         sliding_len_test,
                                         combine,
                                         batch_size_test,
                                         key2dict,
                                         tagger,
                                         collate_fn,
                                         max_char_num_in_tok,
                                         )
        filename2test_data_loader[filename] = test_dataloader

    # if stage == "train":
    train_dataloader = get_dataloader(train_data,
                                      language,
                                      "train",
                                      token_level,
                                      max_seq_len_train,
                                      sliding_len_train,
                                      combine,
                                      batch_size_train,
                                      key2dict,
                                      tagger,
                                      collate_fn,
                                      max_char_num_in_tok,
                                      )
    valid_dataloader = get_dataloader(valid_data,
                                      language,
                                      "valid",
                                      token_level,
                                      max_seq_len_valid,
                                      sliding_len_valid,
                                      combine,
                                      batch_size_valid,
                                      key2dict,
                                      tagger,
                                      collate_fn,
                                      max_char_num_in_tok,
                                      )
    # debug: checking tagging and decoding
    if check_tagging_n_decoding:
        # for checking, take valid data as train data, do additional preprocessing to get tags
        # but take original valid data as golden dataset to evaluate
        ori_data4checking = copy.deepcopy(data4checking)
        data4checking = additional_preprocess(data4checking)
        dataloader4checking = get_dataloader(data4checking,
                                             language,
                                             "train",  # only train data will save the annotations
                                             token_level,
                                             max_seq_len_test,
                                             sliding_len_test,
                                             combine,
                                             batch_size_test,
                                             key2dict,
                                             tagger,
                                             collate_fn,
                                             max_char_num_in_tok,
                                             )
        debug_score_dict, debug_error_analysis_dict = evaluator.check_tagging_n_decoding(dataloader4checking, ori_data4checking)
        json.dump(debug_error_analysis_dict,
                  open(os.path.join(dir_to_save_model, "debug_error_analysis_dict"), "w", encoding="utf-8"),
                  ensure_ascii=False
                  )
        pprint(debug_score_dict)

    # load pretrained model
    if model_state_dict_path is not None:
        model.load_state_dict(torch.load(model_state_dict_path))
        print("model state loaded: {}".format("/".join(model_state_dict_path.split("/")[-2:])))

    # optimizer
    optimizer_class_name = getattr(torch.optim, optimizer_config["class_name"])
    optimizer = optimizer_class_name(model.parameters(), lr=float(lr), **optimizer_config["parameters"])

    trainer = Trainer(run_id, model, train_dataloader, device, optimizer, trainer_config, logger)

    # train and valid
    score_dict4comparing = {}
    for ep in range(epochs):
        # train
        trainer.train(ep, epochs)
        # valid
        pred_samples = evaluator.predict(valid_dataloader, valid_data)
        score_dict, _ = evaluator.score(pred_samples, valid_data, "val")
        logger.log(score_dict)
        dataset2score_dict = {
            "valid_data.json": score_dict,
        }

        for metric_key, current_val_score in score_dict.items():
            if "f1" not in metric_key:
                continue

            if metric_key not in score_dict4comparing:
                score_dict4comparing[metric_key] = {
                    "current": 0.0,
                    "best": 0.0,
                }
            score_dict4comparing[metric_key]["current"] = current_val_score
            score_dict4comparing[metric_key]["best"] = max(current_val_score,
                                                           score_dict4comparing[metric_key]["best"])

            # save models
            if current_val_score > 0. and model_bag_size > 0:
                if (metric_pattern2save is not None or metric_pattern2save != "") and \
                        not re.match(metric_pattern2save, metric_key):
                    continue

                dir_to_save_model_this_key = os.path.join(dir_to_save_model, metric_key)
                if not os.path.exists(dir_to_save_model_this_key):
                    os.makedirs(dir_to_save_model_this_key)

                # save model state
                torch.save(model.state_dict(),
                           os.path.join(dir_to_save_model_this_key,
                                        "model_state_dict_{}_{:.5}.pt".format(ep, current_val_score * 100)))

                # all state paths
                model_state_path_list = glob("{}/model_state_*".format(dir_to_save_model_this_key))
                # sorted by scores
                sorted_model_state_path_list = sorted(model_state_path_list,
                                                      key=get_score_fr_path)
                # only save <model_bag_size> model states
                if len(sorted_model_state_path_list) > model_bag_size:
                    os.remove(sorted_model_state_path_list[0])  # remove the state dict with the minimum score

        # test
        for filename, test_data_loader in filename2test_data_loader.items():
            gold_test_data = filename2test_data[filename]
            pred_samples = evaluator.predict(test_data_loader, gold_test_data)
            score_dict, _ = evaluator.score(pred_samples, gold_test_data, filename.split(".")[0])
            logger.log(score_dict)
            dataset2score_dict[filename] = score_dict

        pprint(dataset2score_dict)
        pprint(score_dict4comparing)
